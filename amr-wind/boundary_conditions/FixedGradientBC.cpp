#include "amr-wind/boundary_conditions/FixedGradientBC.H"
#include "amr-wind/boundary_conditions/bc_utils.H"

namespace amr_wind {

FixedGradientBC::FixedGradientBC(Field& field, amrex::Orientation ori)
    : m_field(field), m_ori(ori)
{}

void FixedGradientBC::apply_bc(const int lev, amrex::MultiFab& mfab, const amrex::Real)
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
                    for (int n=0; n < ncomp; ++n)
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

void FixedGradientBC::operator()(Field& field, const FieldState)
{
    const amrex::Real unused = 0.0;
    const int nlevels = field.repo().num_active_levels();
    for (int lev=0; lev < nlevels; ++lev) {
        apply_bc(lev, field(lev), unused);
    }
}

} // namespace amr_wind
