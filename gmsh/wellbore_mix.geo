
////////////////////////////////////////////////////////////////
// 2D wellbore with circular region
// Created 28/05/2018 by Manouchehr Sanei and Omar Duran
// Labmec, State University of Campinas, Brazil
////////////////////////////////////////////////////////////////

IsquadQ = 1;

Mesh.ElementOrder = 1;
Mesh.SecondOrderLinear = 0;

wr = 0.1;
fr = 4.0;

LargeCBurdenQ = 1;

If(LargeCBurdenQ)
	nt = 5; //usar numeros impares para garantir a simetria
	nr = 5;
	radial_progression = 1.175; // 32
	//radial_progression = 1.075; // 64
	//radial_progression = 1.04; // 128
	//radial_progression = 1.0175; // 256
	//radial_progression = 1.009; // 512
Else
	nt = 2;
	nr = 2;
	radial_progression = 10.25;	
EndIf

// center point
pc = newp; Point(pc) = {0,0,0};

// internal quadrilateral
pqi_1 = newp; Point(pqi_1) = {wr+0.5,wr+0.5,0};
pqi_2 = newp; Point(pqi_2) = {-wr-0.5,wr+0.5,0};
pqi_3 = newp; Point(pqi_3) = {-wr-0.5,-wr-0.5,0};
pqi_4 = newp; Point(pqi_4) = {wr+0.5,-wr-0.5,0};

lqi_1 = newl; Line(lqi_1) = {pqi_1,pqi_2};
lqi_2 = newl; Line(lqi_2) = {pqi_2,pqi_3};
lqi_3 = newl; Line(lqi_3) = {pqi_3,pqi_4};
lqi_4 = newl; Line(lqi_4) = {pqi_4,pqi_1};

// external quadrilateral
pqe_1 = newp; Point(pqe_1) = {fr*0.70710678118,fr*0.70710678118,0};
pqe_2 = newp; Point(pqe_2) = {-fr*0.70710678118,fr*0.70710678118,0};
pqe_3 = newp; Point(pqe_3) = {-fr*0.70710678118,-fr*0.70710678118,0};
pqe_4 = newp; Point(pqe_4) = {fr*0.70710678118,-fr*0.70710678118,0};

lqe_1 = newl; Line(lqe_1) = {pqe_1,pqe_2};
lqe_2 = newl; Line(lqe_2) = {pqe_2,pqe_3};
lqe_3 = newl; Line(lqe_3) = {pqe_3,pqe_4};
lqe_4 = newl; Line(lqe_4) = {pqe_4,pqe_1};

// internal circle
pi_1 = newp; Point(pi_1) = {wr*0.70710678118,wr*0.70710678118,0};
pi_2 = newp; Point(pi_2) = {-wr*0.70710678118,wr*0.70710678118,0};
pi_3 = newp; Point(pi_3) = {-wr*0.70710678118,-wr*0.70710678118,0};
pi_4 = newp; Point(pi_4) = {wr*0.70710678118,-wr*0.70710678118,0};

li_1 = newl; Circle(li_1) = {pi_1,pc,pi_2};
li_2 = newl; Circle(li_2) = {pi_2,pc,pi_3};
li_3 = newl; Circle(li_3) = {pi_3,pc,pi_4};
li_4 = newl; Circle(li_4) = {pi_4,pc,pi_1};

i_circle[] = {li_1,li_2,li_3,li_4};

// external circle
pe_1 = newp; Point(pe_1) = {fr,0,0};
pe_2 = newp; Point(pe_2) = {0,fr,0};
pe_3 = newp; Point(pe_3) = {-fr,0,0};
pe_4 = newp; Point(pe_4) = {0,-fr,0};

le_1 = newl; Circle(le_1) = {pqe_1,pc,pe_2};
le_2 = newl; Circle(le_2) = {pe_2,pc,pqe_2};
le_3 = newl; Circle(le_3) = {pqe_2,pc,pe_3};
le_4 = newl; Circle(le_4) = {pe_3,pc,pqe_3};
le_5 = newl; Circle(le_5) = {pqe_3,pc,pe_4};
le_6 = newl; Circle(le_6) = {pe_4,pc,pqe_4};
le_7 = newl; Circle(le_7) = {pqe_4,pc,pe_1};
le_8 = newl; Circle(le_8) = {pe_1,pc,pqe_1};


e_circle[] = {le_1,le_2,le_3,le_4,le_5,le_6,le_7,le_8};

// Auxiliary geometrical entities
ld1 = newl; Line(ld1) = {pqi_1,pqe_1};
ld2 = newl; Line(ld2) = {pqi_2,pqe_2};
ld3 = newl; Line(ld3) = {pqi_3,pqe_3};
ld4 = newl; Line(ld4) = {pqi_4,pqe_4};

llq_1 = newll; Line Loop(llq_1) = {le_1,le_2,-lqe_1};
llq_2 = newll; Line Loop(llq_2) = {le_3,le_4,-lqe_2};
llq_3 = newll; Line Loop(llq_3) = {le_5,le_6,-lqe_3};
llq_4 = newll; Line Loop(llq_4) = {le_7,le_8,-lqe_4};

llq_5 = newll; Line Loop(llq_5) = {ld1,lqe_1,-ld2,-lqi_1};
llq_6 = newll; Line Loop(llq_6) = {ld2,lqe_2,-ld3,-lqi_2};
llq_7 = newll; Line Loop(llq_7) = {ld3,lqe_3,-ld4,-lqi_3};
llq_8 = newll; Line Loop(llq_8) = {ld4,lqe_4,-ld1,-lqi_4};

llq_9 = newll; Line Loop(llq_9) = {lqi_1,lqi_2,lqi_3,lqi_4};
llq_10 = newll; Line Loop(llq_10) = {li_1,li_2,li_3,li_4};

s1 = news; Plane Surface(s1) = {llq_1};
s2 = news; Plane Surface(s2) = {llq_2};
s3 = news; Plane Surface(s3) = {llq_3};
s4 = news; Plane Surface(s4) = {llq_4};

s5 = news; Plane Surface(s5) = {llq_5};
s6 = news; Plane Surface(s6) = {llq_6};
s7 = news; Plane Surface(s7) = {llq_7};
s8 = news; Plane Surface(s8) = {llq_8};

s9 = news; Plane Surface(s9) = {llq_9,llq_10};

the_circle[] = {s1,s2,s3,s4,s5,s6,s7,s8,s9};

fixed_y_points[]={pe_1,pe_3};
fixed_x_points[]={pe_2,pe_4};

Point{fixed_y_points[],fixed_x_points[]} In Surface{s5};
Point{fixed_y_points[],fixed_x_points[]} In Surface{s6};
Point{fixed_y_points[],fixed_x_points[]} In Surface{s7};
Point{fixed_y_points[],fixed_x_points[]} In Surface{s8};

radial_lines[] = {ld1,ld2,ld3,ld4,lqi_1,lqi_2,lqi_3,lqi_4,lqe_1,lqe_2,lqe_3,lqe_4};
azimuthal_lines[] = {i_circle[],e_circle[]};

Transfinite Line {azimuthal_lines[]} = nt;

Transfinite Line {azimuthal_lines[]} = nt;
Transfinite Line {radial_lines[]} = nr Using Progression radial_progression;

Transfinite Line{lqe_1,ld1,ld2,lqi_1} = nt;
Transfinite Line{lqe_2,ld2,ld3,lqi_2} = nt;
Transfinite Line{lqe_3,ld3,ld4,lqi_3} = nt;
Transfinite Line{lqe_4,ld4,ld1,lqi_4} = nt;

Transfinite Surface{s5,s6,s7,s8};
 If(IsquadQ)
  Recombine Surface{s5,s6,s7,s8};
 EndIf

Physical Surface("Omega") = {the_circle[]};
Physical Line("bc_wellbore") = {i_circle[]};
Physical Line("bc_farfield") = {e_circle[]};
Physical Point("fixed_x") = {fixed_x_points[]};
Physical Point("fixed_y") = {fixed_y_points[]};


Coherence Mesh;

