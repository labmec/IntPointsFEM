//
//  TElastoPlasticData.cpp
//  IntegrationPointExperiments
//
//  Created by Omar Durán on 3/26/19.
//

#include "TElastoPlasticData.h"


TElastoPlasticData::TElastoPlasticData(){
    m_gamma_data.resize(0);
}

TElastoPlasticData::TElastoPlasticData(const TElastoPlasticData &  other){
    
    m_gamma_data    = other.m_gamma_data;
    m_id            = other.m_id;
    m_LER           = other.m_LER;
    m_MC_phi        = other.m_MC_phi;
    m_MC_c          = other.m_MC_c;
    
}

TElastoPlasticData & TElastoPlasticData::operator=(const TElastoPlasticData &  other){
    /// check for self-assignment
    if(&other == this){
        return *this;
    }
    
    m_gamma_data    = other.m_gamma_data;
    m_id            = other.m_id;
    m_LER           = other.m_LER;
    m_MC_phi        = other.m_MC_phi;
    m_MC_c          = other.m_MC_c;
    
    return *this;
}

TElastoPlasticData::~TElastoPlasticData(){
    
}
