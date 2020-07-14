#ifndef LINEARINFLOW_H
#define LINEARINFLOW_H

#include "amr-wind/boundary_conditions/InflowBC.H"

namespace amr_wind {

class LinearInflow : public InflowSpec::Register<LinearInflow>
{
public:
    static std::string identifier() { return "LinearInflow"; }

    LinearInflow(Field& field);

    void init(amrex::Orientation ori, const std::string& pp_namespace) override;
    void
    apply_bc(const int lev, amrex::MultiFab& mfab, const amrex::Real) override;
    void operator()(Field& field, const FieldState) override;

private:
    Field& m_field;
    amrex::Orientation m_ori;

    int m_line_dir{2};
    amrex::Real m_xstart{0.0};
    amrex::Gpu::DeviceVector<amrex::Real> m_start;
    amrex::Gpu::DeviceVector<amrex::Real> m_slope;
};

} // namespace amr_wind

#endif /* LINEARINFLOW_H */
