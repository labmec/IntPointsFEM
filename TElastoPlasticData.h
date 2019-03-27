//
//  TElastoPlasticData.h
//  IntegrationPointExperiments
//
//  Created by Omar Durán on 3/26/19.
//

#ifndef TElastoPlasticData_h
#define TElastoPlasticData_h

#include <stdio.h>
#include "TBCData.h"
#include "TPZElasticResponse.h"

class TElastoPlasticData {
    
public:
    
    /// Stands for boundary data
    std::vector<TBCData> m_gamma_data;
    
    /// material identifier
    int m_id = -1;
    
    TPZElasticResponse m_LER;
    
    /// Friction angle
    REAL m_MC_phi = -1;
    
    /// Cohesion
    REAL m_MC_c = -1;
    
    /// Default constructor
    TElastoPlasticData();
    
    /// Copy constructor
    TElastoPlasticData(const TElastoPlasticData &  other);
    
    /// Assignmet constructor
    TElastoPlasticData & operator=(const TElastoPlasticData &  other);
    
    /// Default destructor
    ~TElastoPlasticData();
    
    
    // @TODO:: NVB please implement access methods
};

#endif /* TElastoPlasticData_h */
