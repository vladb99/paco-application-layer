use ndarray::{Array2, arr2};
use application_layer::job::Job;
use interconnection_network_client::kernel::kernel_functions::KernelFunctions;
fn main() {
    
    let mut size_array = 16;
    let mut kernel:KernelFunctions = KernelFunctions::FnAdd;
    let mut num_of_tasks:usize = 8;

    let matrix_a = arr2(&[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]);

    let matrix_b = arr2(&[
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32],
    ]);

    //let mut matrix_a:Array2<i32>= &[1..(size_array*size_array)]).into_shape((size_array, size_array);
    //let mut matrix_b:Array2<i32> = aview1(&[(size_array*size_array)..(size_array*size_array*2)]).into_shape((size_array, size_array)).unwrap();

    let result:Array2<i32> = Job::create( &matrix_a, &matrix_b, num_of_tasks, kernel);
    println!("first calculation :{} * {} = {}", &matrix_a, &matrix_b, result);


    // size_array = 64*64;
    // num_of_tasks:usize = 64;
    // matrix_a:Array2<i32> = aview1(&[1..(size_array*size_array)]).into_shape((size_array, size_array)).unwrap();
    // matrix_b:Array2<i32> = aview1(&[(size_array*size_array)..(size_array*size_array*2)]).into_shape((size_array, size_array)).unwrap();

    // let result:Array2<i32> = Job::create( &matrix_a, &matrix_b, num_of_tasks, kernel);
    // println!("third calculation :{} * {} = {}", &matrix_a, &matrix_b, result);


    // size_array = 128*128;
    // num_of_tasks:usize = 128;
    // matrix_a:Array2<i32> = aview1(&[1..(size_array*size_array)]).into_shape((size_array, size_array)).unwrap();
    // matrix_b:Array2<i32> = aview1(&[(size_array*size_array)..(size_array*size_array*2)]).into_shape((size_array, size_array)).unwrap();

    // let result:Array2<i32> = Job::create( &matrix_a, &matrix_b, num_of_tasks, kernel);
    // println!("second calculation :{} * {} = {}", &matrix_a, &matrix_b, result);


    // size_array = 128*128;
    // num_of_tasks:usize = 128*8;
    // matrix_a:Array2<i32> = aview1(&[1..(size_array*size_array)]).into_shape((size_array, size_array)).unwrap();
    // matrix_b:Array2<i32> = aview1(&[(size_array*size_array)..(size_array*size_array*2)]).into_shape((size_array, size_array)).unwrap();

    // let result:Array2<i32> = Job::create( &matrix_a, &matrix_b, num_of_tasks, kernel);
    // println!("third calculation :{} * {} = {}", &matrix_a, &matrix_b, result);

}