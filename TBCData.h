//
//  TBCData.h
//  IntegrationPointExperiments
//
//  Created by Omar Durán on 3/26/19.
//

#ifndef TBCData_h
#define TBCData_h

#include <stdio.h>
#include <vector>
#include "pzreal.h"

class TBCData {
    
public:
    
    /// material identifier
    int m_id = -1;
    
    /// bc type - 0 -> Dirichlet and 1 -> Neumann
    int m_type = -1;
    
    /// The boundary data
    std::vector<REAL> m_value;
    
    /// Default constructor
    TBCData();
    
    /// Copy constructor
    TBCData(const TBCData &  other);
    
    /// Assignmet constructor
    TBCData & operator=(const TBCData &  other);
    
    /// Default destructor
    ~TBCData();
    
    // @TODO:: NVB please implement access methods
    
};

#endif /* TBCData_h */
