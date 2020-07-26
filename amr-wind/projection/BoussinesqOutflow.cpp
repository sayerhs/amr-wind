#include "amr-wind/projection/BoussinesqOutflow.H"
#include "amr-wind/core/FieldRepo.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {

BoussinesqOutflow::BoussinesqOutflow(FieldRepo& repo) : m_repo(repo)
{
    if (repo.mesh().finestLevel() > 0) {
        amrex::Abort("BoussinesqOutflow not implemented for multi-level flows");
    }

    if (repo.field_exists("temperature")) {
        m_temperature = &repo.get_field("temperature");
        m_need_outflow_corr = true;
    }

    if (m_need_outflow_corr) {
        amrex::Vector<amrex::Real> gravity{0.0, 0.0, -9.81};
        amrex::Real ref_theta, ref_beta;
        {
            amrex::ParmParse pp("incflo");
            pp.queryarr("gravity", gravity);
        }
        {
            // FIXME: leakage
            amrex::ParmParse pp("BoussinesqBuoyancy");
            if (pp.contains("reference_temperature")) {
                pp.get("reference_temperature", ref_theta);

                if (pp.contains("thermal_expansion_coeff")) {
                    pp.get("thermal_expansion_coeff", ref_beta);
                } else {
                    ref_beta = 1.0 / ref_theta;
                }

                m_ref_theta = ref_theta;
                m_factor = ref_beta * gravity[AMREX_SPACEDIM - 1] *
                    m_repo.mesh().Geom(0).CellSize(2);
            } else {
                m_need_outflow_corr = false;
            }
        }
    }

    if (m_need_outflow_corr) init_data();
}

BoussinesqOutflow::~BoussinesqOutflow() = default;

void BoussinesqOutflow::init_data()
{
    constexpr int nslices = 4;
    amrex::BoxList bl_phi(amrex::IndexType::TheNodeType());
    amrex::BoxList bl_temp(amrex::IndexType::TheCellType());
    const auto& mesh = m_repo.mesh();
    const auto& geom = mesh.Geom(0);
    auto nd_dom =
        amrex::convert(geom.Domain(), amrex::IntVect::TheNodeVector());

    auto& bctyp = m_temperature->bc_type();

    using Dir = amrex::Direction;
    using Ori = amrex::Orientation;

    if (bctyp[Ori(Dir::x, Ori::low)] == BC::pressure_outflow) {
        const auto box = amrex::bdryLo(nd_dom, 0);
        const int jmin = box.smallEnd(1);
        const int jmax = box.bigEnd(1);

        for (int j = jmin; j <= jmax; j += nslices) {
            auto b = box;
            b.setSmall(1, j);
            b.setBig(1, std::min(j + nslices - 1, jmax));
            bl_phi.push_back(b);
            bl_temp.push_back(
                b.grow(amrex::IntVect(1, 1, 0)).growLo(2, 1).enclosedCells());
        }
    }

    if (bctyp[Ori(Dir::x, Ori::high)] == BC::pressure_outflow) {
        const auto box = amrex::bdryHi(nd_dom, 0);
        const int jmin = box.smallEnd(1);
        const int jmax = box.bigEnd(1);

        for (int j = jmin; j <= jmax; j += nslices) {
            auto b = box;
            b.setSmall(1, j);
            b.setBig(1, std::min(j + nslices - 1, jmax));
            bl_phi.push_back(b);
            bl_temp.push_back(
                b.grow(amrex::IntVect(1, 1, 0)).growLo(2, 1).enclosedCells());
        }
    }

    if (bctyp[Ori(Dir::y, Ori::low)] == BC::pressure_outflow) {
        const auto box = amrex::bdryLo(nd_dom, 1);
        const int jmin = box.smallEnd(0);
        const int jmax = box.bigEnd(0);

        for (int j = jmin; j <= jmax; j += nslices) {
            auto b = box;
            b.setSmall(0, j);
            b.setBig(0, std::min(j + nslices - 1, jmax));
            bl_phi.push_back(b);
            bl_temp.push_back(
                b.grow(amrex::IntVect(1, 1, 0)).growLo(2, 1).enclosedCells());
        }
    }

    if (bctyp[Ori(Dir::y, Ori::high)] == BC::pressure_outflow) {
        const auto box = amrex::bdryHi(nd_dom, 1);
        const int jmin = box.smallEnd(0);
        const int jmax = box.bigEnd(0);

        for (int j = jmin; j <= jmax; j += nslices) {
            auto b = box;
            b.setSmall(0, j);
            b.setBig(0, std::min(j + nslices - 1, jmax));
            bl_phi.push_back(b);
            bl_temp.push_back(
                b.grow(amrex::IntVect(1, 1, 0)).growLo(2, 1).enclosedCells());
        }
    }

    if ((bctyp[Ori(Dir::z, Ori::low)] == BC::pressure_inflow) ||
        (bctyp[Ori(Dir::z, Ori::high)] == BC::pressure_inflow)) {
        amrex::Abort("BoussinesqOutflow not implemented for z-dir outflow");
    }

    amrex::BoxArray ba_phi(std::move(bl_phi));
    amrex::BoxArray ba_temp(std::move(bl_temp));
    amrex::DistributionMapping dm{ba_phi};
    m_phi.define(ba_phi, dm, 1, 0);
    m_temp.define(ba_temp, dm, 1, 0);

    if (m_phi.size() < 1) m_need_outflow_corr = false;
}

void BoussinesqOutflow::operator()(ScratchField& phi, const amrex::Real time)
{
    if (!m_need_outflow_corr) return;

    (*m_temperature).fillpatch(time);
    m_temp.ParallelCopy((*m_temperature)(0), 0, 0, 1, 1, 0);
    const amrex::Real fac = m_factor;
    const amrex::Real T0 = m_ref_theta;

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(m_phi); mfi.isValid(); ++mfi) {
        const auto& bx = mfi.validbox();
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);

        const auto& parr = m_phi.array(mfi);
        const auto& tarr = m_temp.const_array(mfi);

        for (int j = lo.y; j <= hi.y; ++j)
            for (int i = lo.x; i <= hi.x; ++i) {
                const int kmin = lo.z;
                const int kmax = hi.z;
#ifdef AMREX_USE_GPU
                if (amrex::Gpu::inLaunchRegion()) {
                    amrex::Scan::PrefixSum<amrex::Real>(
                        kmax - kmin + 1,
                        [=] AMREX_GPU_DEVICE(int kr) -> amrex::Real {
                            int k = kmax - kr - 1;
                            amrex::Real tavg =
                                0.25 *
                                (tarr(i - 1, j - 1, k) + tarr(i, j - 1, k) +
                                 tarr(i - 1, j, k) + tarr(i, j, k));
                            return fac * (tavg - T0);
                        },
                        [=] AMREX_GPU_DEVICE(int kr, const amrex::Real& x) {
                            parr(i, j, kmax - kr) = x;
                        },
                        amrex::Scan::Type::exclusive);
                } else
#endif
                {
                    parr(i, j, kmax) = 0.0;
                    for (int k = kmax - 1; k >= kmin; --k) {
                        // amrex::Real tavg =
                        //     0.25 * (tarr(i - 1, j - 1, k) + tarr(i, j - 1, k) +
                        //             tarr(i - 1, j, k) + tarr(i, j, k));
                        amrex::Real tavg = tarr(i, j, k);
                        parr(i, j, k) = parr(i, j, k + 1) + fac * (tavg - T0);
                    }
                }
            }
    }

    phi(0).ParallelCopy(m_phi);
}

} // namespace amr_wind
