use std::{cmp::{min, max}, ops::Range, time::Duration};
use ndarray::{Array2, parallel::prelude::*, Zip, s};
use interconnection_network_client::{net::Net, frame::{frame_struct::Frame, message_types::MessageType}, kernel::kernel_functions::KernelFunctions, execute::Execute};
use execution_unit::data::Data;


pub struct Job {
    data: Data,
}

impl Job {
    /// This method is the main method of the jobs. It will take care of array formatting, connecting to the network,
    /// parallelizing and bringing all results together by calling the specific methods.
    /// Returns the Kernel-function applied to the matrices
    ///
    /// # Arguments
    ///
    /// * `matrix_a` - First matrix of the calculation
    /// * `matrix_b` - Second matrix of the calculation
    /// * `num_of_tasks` - Number of tasks that the problem should be divided into
    /// * `kernel` - Operation that should be executed on the matrices
    pub fn create(matrix_a: &Array2<i32>, matrix_b: &Array2<i32>, num_of_tasks: usize, kernel: KernelFunctions) -> Array2<i32> {
        //figure out the maximum dimension of the two arrays and fill them with zeros to match the maximum dimension as a squared matrix
        let max_dimension_a = max(matrix_a.nrows(), matrix_a.ncols());
        let max_dimension_b = max(matrix_b.nrows(), matrix_b.ncols());
        let max_dimension = max(max_dimension_a, max_dimension_b);
        let matrix_a_square = Job::convert_to_square_matrix(&matrix_a, max_dimension);
        let matrix_b_square = Job::convert_to_square_matrix(&matrix_b, max_dimension);
        
        // prepare parallisation by splitting the matrices and checking for available nodes
        let tasks = Job::parallelize(&matrix_a_square, &matrix_b_square, num_of_tasks);
        let data_slices = Job::convert_to_data_slices(tasks);
        let nodes = Job::get_network_information();
        let num_nodes: usize = nodes.len();

        //compute results of splitted matrices in parallel on the cluster
        let result: Vec<Data> = data_slices.par_iter().enumerate().map(|(index, data)|{
            //distribute tasks on different nodes
            let current_node = index % num_nodes;
            println!("sending Data to {}", current_node);
    
            //set data for operation and calculate result on the node
            //wait for the acknowledge before continuing by passing true for block
            nodes[current_node].set_data(Frame::new(index as u64, MessageType::SetData, data.clone()), true);
            nodes[current_node].set_execute(Frame::new((index as u64 + (num_of_tasks as u64)) as u64, MessageType::ExecuteFn, Execute{function: kernel.clone()}), true);          
            
            //create dummy data in case of an error
            let mut node_res: Data = Data::new((u32::MAX,u32::MAX), Array2::<i32>::zeros((1, 1)), Array2::<i32>::zeros((1, 1)));
            //get the result back from the node
            nodes[current_node].get_data(Frame::new((index as u64 + (num_of_tasks  as u64  * 2)) as u64, MessageType::GetData, data.clone()), false);
            match nodes[current_node].wait_for_response(Net::get_data_response, (index as u64 + (num_of_tasks  as u64  * 2)) as u64, Duration::from_secs(10)) {
                Ok(frame) => {
                    println!("Received get response - id: {}, data : {:?}", frame.id, frame.data);
                    node_res = frame.data;
                }
                Err(error) => eprintln!("Error : {}", error),
            };   
             
            println!("get Data from {}", current_node);
            node_res
        }).collect();
        //put the splitted resolution matrices together and return the full calculated resolution matrix
        Job::combine_results(result, matrix_a.nrows(), matrix_b.ncols())
    }

    /// Returns a list of `Data`
    ///
    /// # Arguments
    /// * `tasks` - The tasks into which the problem was divided
    fn convert_to_data_slices(tasks: Vec<((Array2<i32>, Array2<i32>), (u32, u32))>) -> Vec<Data> {
        tasks.into_par_iter()
            .map(|tuple| Data::new(tuple.1, tuple.0.0, tuple.0.1))
            .collect()
    }

    /// Returns a list of nodes
    fn get_network_information() -> Vec<Net> {
        let mut nodes = Vec::new();
        let start_port: u64 = 8000;
        let end_port: u64 = 8010;
        let ip_address = "141.37.160.191";
        let mut node_connections: Vec<(&str, u64)> = vec!();
        for port in start_port..end_port {
            node_connections.push((ip_address, port));
        }

        for (ip, port) in node_connections {
            match Net::new(ip, port, false) {
                Ok(node) => nodes.push(node),
                Err(err) => eprintln!("{}", err),
            }
        }

        return nodes;
    }

    /// Returns a list with tasks. Each task has a matrix from `matrix_a` and `matrix_b` and their respective position in the resolution matrix
    ///
    /// # Arguments
    /// * `matrix_a` - First matrix of the problem
    /// * `matrix_b` - Second matrix of the problem
    /// * `num_of_tasks` - The number of tasks into which the problem is to be divided
    fn parallelize(matrix_a: &Array2<i32>, matrix_b: &Array2<i32>, num_of_tasks: usize) -> Vec<((Array2<i32>, Array2<i32>), (u32, u32))> {
        assert!(matrix_a.is_square() && matrix_b.is_square() && matrix_a.ncols() == matrix_b.ncols(), "Matrix A and B must be quadratic and have the same dimensions!");
        assert_eq!(num_of_tasks % 2, 0, "Requested number of sub matrices must be an even number.");
        assert_eq!(matrix_a.ncols() % (num_of_tasks / 2), 0, "The dimension of the matrix must be divisible by the half the number of sub matrices.");

        let dimension: usize = matrix_a.ncols();
        // Calculates the dimension of one small matrix, in order to divide the problem into `num_of_tasks` tasks
        let sub_matrix_size: usize = dimension / (num_of_tasks / 2);
        // Creates an empty list of `num_of_tasks` tuples with default values
        // Each tuple contains two other tuples. The first tuple is for the small matrices taken from `matrix_a` and `matrix_b`
        // The second tuple is for their respective position
        let mut tasks: Vec<((Array2<i32>, Array2<i32>), (u32, u32))> = vec![((Array2::<i32>::zeros((sub_matrix_size, sub_matrix_size)), Array2::<i32>::zeros((sub_matrix_size, sub_matrix_size))), (0, 0)); num_of_tasks];

        // Calculates the position where each matrix should be in the resolution matrix
        // This is important for the next step, as it can run in parallel
        let mut row: u32 = 0;
        let mut column: u32 = 0;
        for mut task in &mut tasks {
            task.1 = (row, column);
            if column == (dimension - sub_matrix_size) as u32 {
                column = 0;
                row += sub_matrix_size as u32;
            } else {
                column += sub_matrix_size as u32;
            }
        }

        // Parallel step where each task is given a small matrix from `matrix_a` and `matrix_b`
        tasks.into_par_iter()
            .map(|mut tuple| {
                let matrix_a = matrix_a.clone();
                let matrix_b = matrix_b.clone();
                let start_position_row = tuple.1.0;
                let start_position_column = tuple.1.1;

                let mut local_row: usize = 0;
                let mut local_column: usize = 0;
                for row in start_position_row..start_position_row + (sub_matrix_size as u32) {
                    for column in start_position_column..start_position_column + (sub_matrix_size as u32) {
                        tuple.0.0[[local_row, local_column]] = matrix_a[[row as usize, column as usize]].clone();
                        tuple.0.1[[local_row, local_column]] = matrix_b[[row as usize, column as usize]].clone();
                        local_column += 1;
                    }
                    local_column = 0;
                    local_row += 1;
                }
                (tuple.0, tuple.1)
            })
            .collect()
    }

    /// Returns a large matrix composed of many small matrices
    ///
    /// # Arguments
    ///
    /// * `data` - List of small matrices returned by the cluster
    /// * `rows` - Number of rows of the result matrix
    /// * `cols` - Number of columns of the result matrix
    fn combine_results(data: Vec<Data>, rows: usize, cols: usize) -> Array2<i32> {
        let slice_size = data.get(0).unwrap().resolution_matrix.nrows() as i32;
        let data: Vec<Data> = data.clone();
        let mut result: Array2<i32> = Array2::zeros((rows, cols));

        for mut d in data {
            // The index defines the position in the large matrix
            let id = d.index;

            // Calculation of the areas of the window where the matrix should be inserted
            let range_row: Range<i32> = id.0 as i32..min(id.0 as i32 + slice_size, rows as i32);
            let range_column: Range<i32> = id.1 as i32..min(id.1 as i32 + slice_size, cols as i32);

            // Parallel loop over the area
            Zip::indexed(result.slice_mut(s![range_row, range_column]))
                .and(&mut d.resolution_matrix)
                .par_for_each(|_, result_element, data_slice| {
                    *result_element = *data_slice;
                });
        }
        result
    }

    /// Returns a square matrix from a given matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to be converted
    /// * `dimension` - The dimension of the square matrix. If necessary, the square matrix is padded with zeroes
    ///
    /// # Examples
    fn convert_to_square_matrix(matrix: &Array2<i32>, dimension: usize) -> Array2<i32> {
        assert!(dimension >= max(matrix.ncols(), matrix.nrows()), "The new dimension must be greater or equal to the greatest dimension in matrix.");

        let mut square_matrix = Array2::<i32>::zeros((dimension, dimension));
        // Parallel loop through slice and matrix
        Zip::indexed(square_matrix.slice_mut(s![..matrix.nrows(), ..matrix.ncols()]))
            .and(matrix)
            .par_for_each(|_, e, val| {
                *e = *val;
            });
        return square_matrix;
    }

    fn test_matrix_multiplication(data_list: &mut Vec<Data>) {
        for data in data_list {
            //data.resolution_matrix = data.first_matrix.dot(&data.second_matrix);
            data.resolution_matrix = data.first_matrix.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combine_results_1() {
        // Build
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

        let parallelized_data = Job::parallelize(&matrix_a, &matrix_b, 4);
        let mut data = Job::convert_to_data_slices(parallelized_data);
        Job::test_matrix_multiplication(&mut data);

        // Execute
        let result = Job::combine_results(data, 4, 4);

        // Verify
        assert_eq!(result.ncols(), 4);
        assert_eq!(result.ncols(), 4);
        assert_eq!(result, arr2(&[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]));
    }

    #[test]
    fn test_convert_to_square_matrix_1() {
        // Build
        let mut matrix = arr2(&[
            [1, 2, 3],
            [4, 5, 6],
        ]);

        // Execute
        let quadratic_matrix = Job::convert_to_square_matrix(&mut matrix, 3);

        // Verify
        assert_eq!(quadratic_matrix.ncols(), 3);
        assert_eq!(quadratic_matrix.nrows(), 3);
        assert_eq!(quadratic_matrix[[1, 0]], 4);
        assert_eq!(quadratic_matrix[[1, 1]], 5);
        assert_eq!(quadratic_matrix[[1, 2]], 6);
        assert_eq!(quadratic_matrix[[2, 0]], 0);
        assert_eq!(quadratic_matrix[[2, 1]], 0);
        assert_eq!(quadratic_matrix[[2, 2]], 0);
    }

    #[test]
    #[should_panic]
    fn test_convert_to_square_matrix_2() {
        // Build
        let mut matrix = arr2(&[
            [1, 2, 3],
            [4, 5, 6],
        ]);

        // Execute
        let _ = Job::convert_to_square_matrix(&mut matrix, 2);
    }

    #[test]
    fn test_parallelize_1() {
        // Build
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

        // Execute
        let mut tasks = Job::parallelize(&matrix_a, &matrix_b, 4);

        // Verify
        assert_eq!(tasks.len(), 4);

        tasks.sort_by(|a, b| {
            a.1.0.cmp(&b.1.0).then(a.1.1.cmp(&b.1.1))
        });

        assert_eq!(tasks.get(0).unwrap().1, (0, 0));
        assert_eq!(tasks.get(1).unwrap().1, (0, 2));
        assert_eq!(tasks.get(2).unwrap().1, (2, 0));
        assert_eq!(tasks.get(3).unwrap().1, (2, 2));

        assert_eq!(tasks.get(0).unwrap().0.0, arr2(&[
            [1, 2],
            [5, 6],
        ]));
        assert_eq!(tasks.get(1).unwrap().0.0, arr2(&[
            [3, 4],
            [7, 8],
        ]));
        assert_eq!(tasks.get(2).unwrap().0.0, arr2(&[
            [9, 10],
            [13, 14],
        ]));
        assert_eq!(tasks.get(3).unwrap().0.0, arr2(&[
            [11, 12],
            [15, 16],
        ]));

        assert_eq!(tasks.get(0).unwrap().0.1, arr2(&[
            [17, 18],
            [21, 22],
        ]));
        assert_eq!(tasks.get(1).unwrap().0.1, arr2(&[
            [19, 20],
            [23, 24],
        ]));
        assert_eq!(tasks.get(2).unwrap().0.1, arr2(&[
            [25, 26],
            [29, 30],
        ]));
        assert_eq!(tasks.get(3).unwrap().0.1, arr2(&[
            [27, 28],
            [31, 32],
        ]));
    }

    #[test]
    #[should_panic]
    fn test_parallelize_2() {
        // Build
        let matrix_a = arr2(&[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]);

        let matrix_b = arr2(&[
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]);

        // Execute
        let _ = Job::parallelize(&matrix_a, &matrix_b, 4);
    }

    #[test]
    #[should_panic]
    fn test_parallelize_3() {
        // Build
        let matrix_a = arr2(&[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]);

        let matrix_b = arr2(&[
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]);

        // Execute
        let _ = Job::parallelize(&matrix_a, &matrix_b, 4);
    }

    #[test]
    #[should_panic]
    fn test_parallelize_4() {
        // Build
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

        // Execute
        let _ = Job::parallelize(&matrix_a, &matrix_b, 10);
    }

    #[test]
    fn test_convert_to_data_slices_1() {
        // Build
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

        let tasks = Job::parallelize(&matrix_a, &matrix_b, 4);

        // Execute
        let data_slices = Job::convert_to_data_slices(tasks);

        // Verify
        assert_eq!(data_slices.len(), 4);
    }
}
