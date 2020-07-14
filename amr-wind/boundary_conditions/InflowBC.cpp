#include "amr-wind/boundary_conditions/InflowBC.H"
#include "amr-wind/boundary_conditions/bc_utils.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {

ConstantInflow::ConstantInflow(Field& field) : m_field(field) {}

void ConstantInflow::init(amrex::Orientation ori, const std::string&)
{
    m_ori = ori;
}

void ConstantInflow::apply_bc(
    const int lev, amrex::MultiFab& mfab, const amrex::Real)
{
    const auto& repo = m_field.repo();
    const auto bcvals = m_field.bc_values_device();
    const int ncomp = m_field.num_comp();
    const int idx = static_cast<int>(m_ori);
    const int idim = m_ori.coordDir();
    const auto islow = m_ori.isLow();
    const auto ishigh = m_ori.isHigh();

    const auto& domain = repo.mesh().Geom(lev).Domain();

    amrex::MFItInfo mfi_info{};
    if (amrex::Gpu::notInLaunchRegion()) mfi_info.SetDynamic(true);
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mfab, mfi_info); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        const auto& bc_a = mfab.array(mfi);

        if (islow && (bx.smallEnd(idim) == domain.smallEnd(idim))) {
            amrex::ParallelFor(
                bc_utils::lower_boundary_faces(bx, idim),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    for (int n = 0; n < ncomp; ++n)
                        bc_a(i, j, k, n) = bcvals[idx][n];
                });
        }

        if (ishigh && (bx.bigEnd(idim) == domain.bigEnd(idim))) {
            amrex::ParallelFor(
                amrex::bdryHi(bx, idim),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    for (int n = 0; n < ncomp; ++n)
                        bc_a(i, j, k, n) = bcvals[idx][n];
                });
        }
    }
}

void ConstantInflow::operator()(Field& field, const FieldState)
{
    const amrex::Real unused = 0.0;
    const int nlevels = field.repo().num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        apply_bc(lev, field(lev), unused);
    }
}

InflowBC::InflowBC(
    Field& field, amrex::Orientation ori, const std::string& pp_namespace)
{
    amrex::ParmParse pp(pp_namespace);
    std::string itype("ConstantInflow");
    const std::string key = field.name() + "/inflow_type";
    pp.query(key.c_str(), itype);

    m_bc_impl = InflowSpec::create(itype, field);
    m_bc_impl->init(ori, pp_namespace);
}

void InflowBC::apply_bc(
    const int lev, amrex::MultiFab& mfab, const amrex::Real time)
{
    m_bc_impl->apply_bc(lev, mfab, time);
}

void InflowBC::operator()(Field& field, const FieldState fstate)
{
    (*m_bc_impl)(field, fstate);
}

} // namespace amr_wind
