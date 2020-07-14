#ifndef FIXEDGRADIENTBC_H
#define FIXEDGRADIENTBC_H

#include "amr-wind/core/FieldBCOps.H"
#include "amr-wind/core/FieldRepo.H"

#include "AMReX_Orientation.H"

namespace amr_wind {

/** Applies a non-zero gradient at the domain boundary
 *  \ingroup field_bc
 *
 *  Used with the inhomogeneous Neumann linear solver BC to apply a
 *  user-specified fixed gradient at the domain boundary.
 */
class FixedGradientBC : public FieldBCIface
{
public:
    FixedGradientBC(Field& field, amrex::Orientation ori);

    virtual void apply_bc(
        const int lev, amrex::MultiFab& mfab, const amrex::Real time) override;

    void operator()(Field& field, const FieldState) override;

private:
    Field& m_field;

    amrex::Orientation m_ori;
};

}

#endif /* FIXEDGRADIENTBC_H */
