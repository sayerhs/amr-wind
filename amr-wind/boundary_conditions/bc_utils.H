#ifndef BC_UTILS_H
#define BC_UTILS_H

#include "AMReX_Box.H"

namespace amr_wind {
namespace bc_utils {

AMREX_FORCE_INLINE
amrex::Box lower_boundary_faces(const amrex::Box& b, int dir)
{
    amrex::IntVect lo(b.smallEnd());
    amrex::IntVect hi(b.bigEnd());
    int sm = lo[dir];
    lo.setVal(dir, sm - 1);
    hi.setVal(dir, sm - 1);
    amrex::IndexType bxtype(b.ixType());
    bxtype.set(dir);
    return amrex::Box(lo, hi, bxtype);
}

} // namespace bc_utils
} // namespace amr_wind

#endif /* BC_UTILS_H */
