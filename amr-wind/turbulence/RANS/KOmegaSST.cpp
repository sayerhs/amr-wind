#include "amr-wind/turbulence/RANS/KOmegaSST.H"
#include "amr-wind/equation_systems/PDEBase.H"
#include "amr-wind/turbulence/TurbModelDefs.H"
#include "amr-wind/fvm/gradient.H"
#include "amr-wind/fvm/strainrate.H"
#include "amr-wind/turbulence/turb_utils.H"
#include "amr-wind/equation_systems/tke/TKE.H"
#include "amr-wind/equation_systems/sdr/SDR.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {
namespace turbulence {

template <typename Transport>
KOmegaSST<Transport>::KOmegaSST(CFDSim& sim)
    : TurbModelBase<Transport>(sim), m_vel(sim.repo().get_field("velocity")),
      m_f1(sim.repo().declare_field("f1",1,1,1)),
      m_shear_prod(sim.repo().declare_field("shear_prod",1, 1, 1)),
      m_diss(sim.repo().declare_field("dissipation",1, 1, 1)),
      m_sdr_src(sim.repo().declare_field("omega_src",1,1,1)),
      m_rho(sim.repo().get_field("density")),
      m_walldist(sim.repo().get_field("wall_dist"))
{
    //auto& tke_eqn = sim.pde_manager().register_transport_pde(pde::TKE::pde_name());
    //m_tke = &(tke_eqn.fields().field);
    m_tke = &(sim.repo().get_field("tke"));
    
    //auto& sdr_eqn = sim.pde_manager().register_transport_pde(pde::SDR::pde_name());
    //m_sdr = &(sdr_eqn.fields().field);
    m_sdr = &(sim.repo().get_field("sdr"));
   
    {
        const std::string coeffs_dict = this->model_name() + "_coeffs";
        amrex::ParmParse pp(coeffs_dict);
        pp.query("beta_star", this->m_beta_star);
        pp.query("alpha1", this->m_alpha1);
        pp.query("alpha2", this->m_alpha2);
        pp.query("beta1", this->m_beta1);
        pp.query("beta2", this->m_beta2);
        pp.query("sigma_k1", this->m_sigma_k1);
        pp.query("sigma_k2", this->m_sigma_k2);
        pp.query("sigma_omega1", this->m_sigma_omega1);
        pp.query("sigma_omega2", this->m_sigma_omega2);
    }

    // TKE source term to be added to PDE
    turb_utils::inject_turbulence_src_terms(pde::TKE::pde_name(), {"TKERANSSrc"});
    
}

template <typename Transport>
KOmegaSST<Transport>::~KOmegaSST() = default;

template <typename Transport>
TurbulenceModel::CoeffsDictType KOmegaSST<Transport>::model_coeffs() const
{
    return TurbulenceModel::CoeffsDictType{
        {"beta_star", this->m_beta_star},
        {"alpha1", this->m_alpha1},
        {"alpha2", this->m_alpha2},
        {"beta1", this->m_beta1},
        {"beta2", this->m_beta2},
        {"sigma_k1", this->m_sigma_k1},
        {"sigma_k2", this->m_sigma_k2},
        {"sigma_omega1", this->m_sigma_omega1},
        {"sigma_omega2", this->m_sigma_omega2},
        {"a1", this->m_a1}};
}

template <typename Transport>
void KOmegaSST<Transport>::update_turbulent_viscosity(
    const FieldState fstate)
{
    BL_PROFILE("amr-wind::" + this->identifier() + "::update_turbulent_viscosity")

        /* Steps
        1. Calculate CDkw - Needs gradK, gradOmega
        2. Calculate F1 - Needs CDkw
        3. Calculate alpha, beta - Needs F1
        4. Calculate F2
        5. Calculate S
        6. Calculate mut - Needs S, F2
        7. Calculate TKE Shear production term Pk - Needs mut
        8. Calculate TKE dissipation term 
        9. Calculate SDR source terms - Needs alpha, beta, F1, gradK, gradOmega
        */

    amrex::Print() << "Calling update_turbulent_viscosity " << std::endl;

    amrex::Print() << "Ghost cells for tke is " << m_tke->num_grow() << std::endl;
    amrex::Print() << "Ghost cells for sdr is " << m_sdr->num_grow() << std::endl;
    
    m_tke->fillpatch(this->m_sim.time().current_time());
    m_sdr->fillpatch(this->m_sim.time().current_time());
    
    auto gradK = (this->m_sim.repo()).create_scratch_field(3,0);
    fvm::gradient(*gradK, m_tke->state(fstate) );

    auto gradOmega = (this->m_sim.repo()).create_scratch_field(3,0);
    fvm::gradient(*gradOmega, m_sdr->state(fstate) );
    
    auto& vel = this->m_vel.state(fstate);
    // Compute strain rate into shear production term
    fvm::strainrate(this->m_shear_prod, vel);
    
    auto& mu_turb = this->mu_turb();
    const amrex::Real lam_mu = (this->m_transport).viscosity();
    const amrex::Real beta_star = this->m_beta_star;
    const amrex::Real alpha1 = this->m_alpha1;
    const amrex::Real alpha2 = this->m_alpha2;
    const amrex::Real beta1 = this->m_beta1;
    const amrex::Real beta2 = this->m_beta2;
    const amrex::Real sigma_omega2 = this->m_sigma_omega2;
    const amrex::Real a1 = this->m_a1;
    
    auto& den = this->m_rho.state(fstate);
    auto& repo = mu_turb.repo();

    const int nlevels = repo.num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {

        for (amrex::MFIter mfi(mu_turb(lev)); mfi.isValid(); ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& mu_arr = mu_turb(lev).array(mfi);
            const auto& rho_arr = den(lev).const_array(mfi);
            const auto& gradK_arr = (*gradK)(lev).array(mfi);
            const auto& gradOmega_arr = (*gradOmega)(lev).array(mfi);
            const auto& tke_arr = (*this->m_tke)(lev).array(mfi);
            const auto& sdr_arr = (*this->m_sdr)(lev).array(mfi);
            const auto& wd_arr = (this->m_walldist)(lev).array(mfi);            
            const auto& shear_prod_arr = (this->m_shear_prod)(lev).array(mfi);
            const auto& diss_arr = (this->m_diss)(lev).array(mfi);
            const auto& sdr_src_arr = (this->m_sdr_src)(lev).array(mfi);
            const auto& f1_arr = (this->m_f1)(lev).array(mfi);
            
            amrex::ParallelFor(
                bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                  amrex::Real gko =
                      -(gradK_arr(i,j,k,0) * gradOmega_arr(i,j,k,0)
                        + gradK_arr(i,j,k,1) * gradOmega_arr(i,j,k,1)
                        + gradK_arr(i,j,k,2) * gradOmega_arr(i,j,k,2));
                  

                  amrex::Real cdkomega = std::max(1e-10,2.0 * rho_arr(i,j,k) * sigma_omega2 * gko / std::max(sdr_arr(i,j,k),1e-15) );
                  amrex::Real tmp1 = 4.0 * rho_arr(i,j,k) * sigma_omega2 * tke_arr(i,j,k) / (cdkomega * wd_arr(i,j,k));
                  amrex::Real tmp2 = std::sqrt(tke_arr(i,j,k))/(beta_star * sdr_arr(i,j,k)*wd_arr(i,j,k) + 1e-15);
                  amrex::Real tmp3 = 500.0*lam_mu/(wd_arr(i,j,k)*wd_arr(i,j,k)*sdr_arr(i,j,k)*rho_arr(i,j,k) + 1e-15);
                  amrex::Real tmp4 = shear_prod_arr(i,j,k);

                  amrex::Real tmp_f1 = std::tanh(std::min( std::max(tmp2,tmp3), tmp1));

                  amrex::Real alpha = tmp_f1 * (alpha1 - alpha2) + alpha2;
                  amrex::Real beta = tmp_f1 * (beta1 - beta2) + beta2;

                  amrex::Real f2 = std::tanh(std::pow(std::max(2.0 * tmp2, tmp3),2));
                  mu_arr(i, j, k) = rho_arr(i,j,k) * a1 * tke_arr(i,j,k) /
                      std::max(a1 * sdr_arr(i,j,k), tmp4 * f2);
                  if ( mu_arr(i,j,k) < 1e-6)
                      std::cerr << "mu_turb = " << mu_arr(i,j,k) << ", shear_prod = " << tmp4
                                << ", tke = " << tke_arr(i,j,k) << ", omega = " << sdr_arr(i,j,k) << std::endl;

                  f1_arr(i,j,k) = tmp_f1;
                  
                  diss_arr(i,j,k) = - beta_star * rho_arr(i,j,k) * tke_arr(i,j,k) * sdr_arr(i,j,k);

                  sdr_src_arr(i,j,k) = rho_arr(i,j,k)
                      * (alpha * tmp4 * tmp4
                         - beta * sdr_arr(i,j,k) * sdr_arr(i,j,k)
                         + 2.0 * (1-tmp_f1) * sigma_omega2 * gko / (sdr_arr(i,j,k) + 1e-15) );

                  shear_prod_arr(i,j,k) = std::max(mu_arr(i,j,k) * tmp4 * tmp4,
                                                   10.0 * beta_star * rho_arr(i,j,k)
                                                   * tke_arr(i,j,k) * sdr_arr(i,j,k));
            });
        }
    }

    mu_turb.fillpatch(this->m_sim.time().current_time());
    
}

template <typename Transport>
void KOmegaSST<Transport>::update_scalar_diff(
    Field& deff, const std::string& name)
{

    BL_PROFILE("amr-wind::" + this->identifier() + "::update_scalar_diff");

    const amrex::Real lam_mu = (this->m_transport).viscosity();
    auto& mu_turb = this->mu_turb();

    amrex::Print() << " scalar diff name = " << name << std::endl;
    
    if (name == "tke") {
        const amrex::Real sigma_k1 = this->m_sigma_k1;
        const amrex::Real sigma_k2 = this->m_sigma_k2;
        auto& repo = deff.repo();
        const int nlevels = repo.num_active_levels();
        for (int lev=0; lev < nlevels; ++lev) {
            for (amrex::MFIter mfi(deff(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                const auto& mu_arr = mu_turb(lev).array(mfi);
                const auto& f1_arr = (this->m_f1)(lev).array(mfi);
                const auto& deff_arr = deff(lev).array(mfi);
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                    deff_arr(i,j,k) = lam_mu
                        + (f1_arr(i,j,k)*(sigma_k1-sigma_k2) + sigma_k2)*mu_arr(i,j,k);
                });
            }
        }
        
    } else if (name == "sdr") {
        const amrex::Real sigma_omega1 = this->m_sigma_omega1;
        const amrex::Real sigma_omega2 = this->m_sigma_omega2;
        auto& repo = deff.repo();
        const int nlevels = repo.num_active_levels();
        for (int lev=0; lev < nlevels; ++lev) {
            for (amrex::MFIter mfi(deff(lev)); mfi.isValid(); ++mfi) {
                const auto& bx = mfi.tilebox();
                const auto& mu_arr = mu_turb(lev).array(mfi);
                const auto& f1_arr = (this->m_f1)(lev).array(mfi);
                const auto& deff_arr = deff(lev).array(mfi);
                amrex::ParallelFor(
                    bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

                    deff_arr(i,j,k) = lam_mu
                        + (f1_arr(i,j,k)*(sigma_omega1-sigma_omega2)
                           + sigma_omega2)*mu_arr(i,j,k);
                });
            }
        }
        
    }

    deff.fillpatch(this->m_sim.time().current_time());    
}


} // namespace turbulence

INSTANTIATE_TURBULENCE_MODEL(KOmegaSST);

} // namespace amr_wind
