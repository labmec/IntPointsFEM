//
//  TBCData.cpp
//  IntegrationPointExperiments
//
//  Created by Omar Durán on 3/26/19.
//

#include "TBCData.h"

TBCData::TBCData(){
    m_value.resize(0);
}

TBCData::TBCData(const TBCData &  other){
    
    m_id    = other.m_id;
    m_type  = other.m_type;
    m_value = other.m_value;
}

TBCData & TBCData::operator=(const TBCData &  other){
    
    /// check for self-assignment
    if(&other == this){
        return *this;
    }
    
    m_id    = other.m_id;
    m_type  = other.m_type;
    m_value = other.m_value;
    
    return *this;
}

TBCData::~TBCData(){
    
}
