#!/bin/bash
test_type="GPU_SHARED"

cd ../../IntPointsFEM-build

rm -rf ${test_type}
mkdir ${test_type}

p_order="1"
echo ""
echo Test type:     ${test_type}
echo Polynomial order:  ${p_order}  
cmake -DCMAKE_BUILD_TYPE=Release -DUSING_CUDA=on -DUSING_TBB=on -DUSING_SPARSE=on -DO_LINEAR=on -DO_QUADRATIC=off -DO_CUBIC=off -DCOMPUTE_WITH_MODIFIED=off -DCOMPUTE_WITH_PZ=off -DUSE_SHARED=on ../IntPointsFEM > any.txt

make -j32 > any.txt

for mesh_id in 1 2 3 4 5
do
    echo ""
    echo "Mesh id:      $mesh_id"
    mkdir ${test_type}/order${p_order}-mesh$mesh_id
    for i in 1 2 3 4 5 
    do
        echo $i of 5
        ./IntPointsFEM $mesh_id > ${test_type}/order${p_order}-mesh$mesh_id/out-$i.txt
    done
done

p_order="2"
echo ""
echo Test type:     ${test_type}
echo Polynomial order:  ${p_order}  
cmake -DCMAKE_BUILD_TYPE=Release -DUSING_CUDA=on -DUSING_TBB=on -DUSING_SPARSE=on -DO_LINEAR=off -DO_QUADRATIC=on -DO_CUBIC=off -DCOMPUTE_WITH_MODIFIED=off -DCOMPUTE_WITH_PZ=off -DUSE_SHARED=on ../IntPointsFEM > any.txt

make -j32 > any.txt

for mesh_id in 1 2 3 4
do
    echo ""
    echo "Mesh id:      $mesh_id"
    mkdir ${test_type}/order${p_order}-mesh$mesh_id
    for i in 1 2 3 4 5 
    do
        echo $i of 5
        ./IntPointsFEM $mesh_id > ${test_type}/order${p_order}-mesh$mesh_id/out-$i.txt
    done
done

p_order="3"
echo ""
echo Test type:     ${test_type}
echo Polynomial order:  ${p_order}  
cmake -DCMAKE_BUILD_TYPE=Release -DUSING_CUDA=on -DUSING_TBB=on -DUSING_SPARSE=on -DO_LINEAR=off -DO_QUADRATIC=off -DO_CUBIC=on -DCOMPUTE_WITH_MODIFIED=off -DCOMPUTE_WITH_PZ=off -DUSE_SHARED=on ../IntPointsFEM > any.txt

make -j32 > any.txt

for mesh_id in 1 2 3 4 
do
    echo ""
    echo "Mesh id:      $mesh_id"
    mkdir ${test_type}/order${p_order}-mesh$mesh_id
    for i in 1 2 3 4 5 
    do
        echo $i of 5
        ./IntPointsFEM $mesh_id > ${test_type}/order${p_order}-mesh$mesh_id/out-$i.txt
    done
done

rm any.txt


