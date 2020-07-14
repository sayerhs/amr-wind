#ifndef BOUSSINESQOUTFLOW_H
#define BOUSSINESQOUTFLOW_H

#include "AMReX_MultiFab.H"

namespace amr_wind {

class Field;
class FieldRepo;
class ScratchField;

class BoussinesqOutflow
{
public:
    BoussinesqOutflow(FieldRepo& repo);

    ~BoussinesqOutflow();

    void operator()(ScratchField& phi, const amrex::Real time);

private:
    void init_data();

    FieldRepo& m_repo;
    Field* m_temperature{nullptr};

    amrex::MultiFab m_phi;
    amrex::MultiFab m_temp;

    amrex::Real m_ref_theta{0.0};
    amrex::Real m_factor{0.0};

    bool m_need_outflow_corr{false};
};

}

#endif /* BOUSSINESQOUTFLOW_H */
