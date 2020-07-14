#include "amr-wind/boundary_conditions/LinearInflow.H"
#include "amr-wind/boundary_conditions/bc_utils.H"

#include "AMReX_ParmParse.H"

namespace amr_wind {

LinearInflow::LinearInflow(Field& field) : m_field(field) {}

void LinearInflow::init(amrex::Orientation ori, const std::string& pp_namespace)
{
    m_ori = ori;
    const auto ncomp = m_field.num_comp();
    m_start.resize(ncomp);
    m_slope.resize(ncomp);

    {
        const auto fname = m_field.name();

        amrex::ParmParse pp(pp_namespace);
        pp.query((fname + "/line_dir").c_str(), m_line_dir);
        pp.query((fname + "/origin").c_str(), m_xstart);
        amrex::Vector<amrex::Real> buf;

        pp.getarr((fname + "/start").c_str(), buf);
        AMREX_ALWAYS_ASSERT(buf.size() == ncomp);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, buf.begin(), buf.end(), m_start.begin());

        pp.getarr((fname + "/slope").c_str(), buf);
        AMREX_ALWAYS_ASSERT(buf.size() == ncomp);
        amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, buf.begin(), buf.end(), m_slope.begin());
    }
}

void LinearInflow::apply_bc(
    const int lev, amrex::MultiFab& mfab, const amrex::Real)
{
    const auto& repo = m_field.repo();
    const int ncomp = m_field.num_comp();
    const int idim = m_ori.coordDir();
    const auto islow = m_ori.isLow();
    const auto ishigh = m_ori.isHigh();

    const auto& geom = repo.mesh().Geom(lev);
    const auto& domain = geom.Domain();
    const auto problo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const auto offset = m_xstart;
    const int ldir = m_line_dir;
    const auto* zstart = m_start.data();
    const auto* zslope = m_slope.data();

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
                    amrex::Real zco = problo[ldir] + (k + 0.5) * dx[ldir];
                    for (int n = 0; n < ncomp; ++n) {
                        bc_a(i, j, k, n) =
                            zstart[n] + zslope[n] * (zco - offset);
                    }
                });
        }

        if (ishigh && (bx.bigEnd(idim) == domain.bigEnd(idim))) {
            amrex::ParallelFor(
                amrex::bdryHi(bx, idim),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::Real zco = problo[ldir] + (k + 0.5) * dx[ldir];
                    for (int n = 0; n < ncomp; ++n) {
                        bc_a(i, j, k, n) =
                            zstart[n] + zslope[n] * (zco - offset);
                    }
                });
        }
    }
}

void LinearInflow::operator()(Field& field, const FieldState)
{
    const amrex::Real unused = 0.0;
    const int nlevels = field.repo().num_active_levels();
    for (int lev = 0; lev < nlevels; ++lev) {
        apply_bc(lev, field(lev), unused);
    }
}

} // namespace amr_wind
