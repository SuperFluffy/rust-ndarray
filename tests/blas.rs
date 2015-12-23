#![cfg(feature = "rblas")]

extern crate rblas;
extern crate num;
#[macro_use] extern crate ndarray;

use rblas::Gemm;
use rblas::attribute::Transpose;

use num::Float;

use ndarray::{
    OwnedArray,
    ArrayView,
    ArrayViewMut,
    arr2,
    Ix,
    ShapeError
};

use ndarray::blas::AsBlas;

#[test]
fn strided_matrix() {
    // smoke test, a matrix multiplication of uneven size
    let (n, m) = (45, 33);
    let mut a = OwnedArray::linspace(0., ((n * m) - 1) as f32, n as usize * m as usize ).into_shape((n, m)).unwrap();
    let mut b = ndarray::linalg::eye(m);
    let mut res = OwnedArray::zeros(a.dim());
    Gemm::gemm(&1., Transpose::NoTrans, &a.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    assert_eq!(res, a);

    // matrix multiplication, strided
    let mut aprim = a.to_shared();
    aprim.islice(s![0..12, 0..11]);
    println!("{:?}", aprim.shape());
    println!("{:?}", aprim.strides());
    let mut b = ndarray::linalg::eye(aprim.shape()[1]);
    let mut res = OwnedArray::zeros(aprim.dim());
    Gemm::gemm(&1., Transpose::NoTrans, &aprim.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    assert_eq!(res, aprim);

    // Transposed matrix multiply
    let (np, mp) = aprim.dim();
    let mut res = OwnedArray::zeros((mp, np));
    let mut b = ndarray::linalg::eye(np);
    Gemm::gemm(&1., Transpose::Trans, &aprim.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    let mut at = aprim.clone();
    at.swap_axes(0, 1);
    assert_eq!(at, res);

    // strided, needs copy
    let mut abis = a.to_shared();
    abis.islice(s![0..12, ..;2]);
    println!("{:?}", abis.shape());
    println!("{:?}", abis.strides());
    let mut b = ndarray::linalg::eye(abis.shape()[1]);
    let mut res = OwnedArray::zeros(abis.dim());
    Gemm::gemm(&1., Transpose::NoTrans, &abis.blas(), Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    assert_eq!(res, abis);
}

#[test]
fn strided_view() {
    // smoke test, a matrix multiplication of uneven size
    let (n, m) = (45, 33);
    let mut a = OwnedArray::linspace(0., ((n * m) - 1) as f32, n as usize * m as usize ).into_shape((n, m)).unwrap();
    let mut b = ndarray::linalg::eye(m);
    let mut res = OwnedArray::zeros(a.dim());
    Gemm::gemm(&1.,
               Transpose::NoTrans, &a.blas_view_mut_checked().unwrap(),
               Transpose::NoTrans, &b.blas_view_mut_checked().unwrap(),
               &0., &mut res.blas_view_mut_checked().unwrap());
    assert_eq!(res, a);

    // matrix multiplication, strided
    let mut a2 = a.clone();
    let aprim = a2.view().slice(s![0..12, 0..11]);
    let mut b = ndarray::linalg::eye(aprim.shape()[1]);
    let mut res = OwnedArray::zeros(aprim.dim());
    Gemm::gemm(&1.,
               Transpose::NoTrans, &aprim.blas_view(),
               Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas());
    assert_eq!(res, aprim);

    // test out with matrices where lower axis is strided but has length 1
    let mut a3 = arr2(&[[1., 2., 3.]]);
    a3.swap_axes(0, 1);
    let mut b = ndarray::linalg::eye(a3.shape()[1]);
    let mut res = arr2(&[[0., 0., 0.]]);
    res.swap_axes(0, 1);
    Gemm::gemm(&1.,
               Transpose::NoTrans, &a3.blas_view_mut(),
               Transpose::NoTrans, &b.blas(),
               &0., &mut res.blas_view_mut());
    assert_eq!(res, a3);
}

#[test]
fn as_blas() {
    let mut a = OwnedArray::<f32, _>::zeros((4, 4));
    assert!(a.blas_view_mut_checked().is_ok());
    a.swap_axes(0, 1);
    assert!(a.blas_view_mut_checked().is_err());
    a.swap_axes(0, 1);

    {
        // increased row stride
        let mut b = a.slice_mut(s![..;2, ..]);
        assert!(b.blas_view_mut_checked().is_ok());
        b.blas_view_mut(); // no panic
    }
    {
        // inner dimension is not contig
        let mut b = a.slice_mut(s![.., ..;2]);
        assert!(b.blas_view_mut_checked().is_err());
    }
    {
        // inner dimension is length 1, is ok again
        let mut b = a.slice_mut(s![.., ..;4]);
        assert!(b.blas_view_mut_checked().is_ok());
        b.blas_view_mut();
    }
}

type Ix2 = (Ix, Ix);
fn dot<F>(mut a: ArrayView<F, Ix2>, mut b: ArrayView<F, Ix2>,
          c: &mut ArrayViewMut<F, Ix2>) -> Result<(), ShapeError>
    where F: Gemm + Float,
{
    let at = Transpose::NoTrans;
    let bt = Transpose::NoTrans;

    let ba = try!(a.blas_view_checked());
    let bb = try!(b.blas_view_checked());
    let mut bc = try!(c.blas_view_mut_checked());
    F::gemm(&F::one(),
            at, &ba,
            bt, &bb,
            &F::zero(), &mut bc);
    Ok(())
}

#[test]
fn test_dot() {
    let mut a = arr2(&[[1., 2.],
                       [0., 3.]]);

    let mut b = a.clone();
    let mut res = a.clone();
    res.assign_scalar(&0.);

    dot(a.view(), b.view(), &mut res.view_mut()).unwrap();
    println!("{:?}", res);

    a.swap_axes(0, 1);
    res.assign_scalar(&0.);

    let result = dot(a.view(), b.view(), &mut res.view_mut());
    assert!(result.is_err());
}
