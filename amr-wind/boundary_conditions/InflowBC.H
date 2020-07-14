#ifndef INFLOWBC_H
#define INFLOWBC_H

#include "amr-wind/core/Factory.H"
#include "amr-wind/core/FieldBCOps.H"
#include "amr-wind/core/FieldRepo.H"

namespace amr_wind {

class InflowSpec : public Factory<InflowSpec, Field&>
{
public:
    static std::string base_identifier() { return "InflowSpec"; }

    virtual void
    init(amrex::Orientation ori, const std::string& pp_namespace) = 0;
    virtual void
    apply_bc(const int lev, amrex::MultiFab& mfab, const amrex::Real) = 0;
    virtual void operator()(Field& field, const FieldState) = 0;
};

class ConstantInflow : public InflowSpec::Register<ConstantInflow>
{
public:
    static std::string identifier() { return "ConstantInflow"; }

    ConstantInflow(Field& field);

    void init(amrex::Orientation ori, const std::string& pp_namespace) override;
    void
    apply_bc(const int lev, amrex::MultiFab& mfab, const amrex::Real) override;
    void operator()(Field& field, const FieldState) override;

private:
    Field& m_field;
    amrex::Orientation m_ori;
};

class InflowBC : public FieldBCIface
{
public:
    InflowBC(
        Field& field, amrex::Orientation ori, const std::string& pp_namespace);

    void apply_bc(const int lev, amrex::MultiFab&, const amrex::Real) override;

    void operator()(Field& field, const FieldState) override;

private:
    std::unique_ptr<InflowSpec> m_bc_impl;
};

} // namespace amr_wind

#endif /* INFLOWBC_H */
