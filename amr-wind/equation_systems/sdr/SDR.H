#ifndef SDR_H
#define SDR_H

#include "amr-wind/equation_systems/PDETraits.H"
#include "amr-wind/equation_systems/SchemeTraits.H"
#include "amr-wind/equation_systems/PDEHelpers.H"
#include "amr-wind/equation_systems/PDE.H"
#include "amr-wind/equation_systems/sdr/SDRSource.H"

namespace amr_wind {
namespace pde {

struct SDR : ScalarTransport
{
    using SrcTerm = SDRSource;

    static std::string pde_name() { return "SDR"; }
    static std::string var_name() { return "sdr"; }
};

template<>
struct TurbulenceOp<SDR>
{
    TurbulenceOp(turbulence::TurbulenceModel& tmodel,
                 PDEFields& fields)
        : m_tmodel(tmodel), m_fields(fields)
    {}

    void operator()()
    {
        auto& mueff = m_fields.mueff;
        m_tmodel.update_scalar_diff(mueff, "sdr");
    }

    turbulence::TurbulenceModel& m_tmodel;
    PDEFields& m_fields;
};

} // namespace pde
} // namespace amr_wind

#endif /* SDR_H */
