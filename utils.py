import h5py


def get_dataset_name(filename_with_dir):
    """
        Get the dataset name from the filename.
        Parameters:
            filename_with_dir (str): The filename with directory.
        Returns:
            dataset_name (str): The name of the dataset.
    """
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('.')[:-1]
    dataset_name = ''.join(temp)
    return dataset_name

def get_dataset_from_h5py(filename_path):
    """
        Get the dataset from the h5py file.
        Parameters:
        filename_path (str): The path to the h5py file.
        Returns:
        matrix (numpy.ndarray): The dataset as a numpy array.
    """

    matrix = None
    try:
        with h5py.File(filename_path, 'r') as f:
            # List all datasets or groups in the file
            avilable_datasets = list(f.keys())
            if len(avilable_datasets) == 0:
                print("No datasets or groups found in the file.")
            else:
                print(f"Available datasets file {filename_path}: {avilable_datasets}")

            dataset_name = avilable_datasets[0]
            # Check if the dataset exists in the file
            if dataset_name in f:
                matrix = f.get(dataset_name)[()]            
            else:
                print(f"Dataset '{dataset_name}' not found in the file.")
    except OSError as e:
        print(f"Error opening file {filename_path}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return matrix


matrix = get_dataset_from_h5py("Data/Final Project data/Final Project data/Intra/train/task_working_memory_105923_3.h5")
if matrix is not None:
    print(f"Matrix shape: {matrix.shape}")