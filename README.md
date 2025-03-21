# AlignMalloc
AlignMalloc is the first repository to to comprehensively address both memory allocation and UVM access performance concurrently.

We analyze the fundamental inefficiencies in UVM access and first reveal the mismatch between memory access and UVM prefetching methods. To address the challenges of UVM mismatches in GPU memory management, we propose AlignMalloc, a novel dynamic allocation system specifically designed to align memory arrangement with the underlying UVM prefetching strategy for large-scale applications. 

If this framework contributes to your research or projects, kindly acknowledge it by citing the relevant papers that align with the functionalities you utilize.

# How to use?
## Results from Nsight
The experimental results were comprehensively documented using Nvidia’s Nsight Systems. The resulting files, labeled with extensions .nsys-rep and .sqlite, are designated for distinct purposes: the former can be accessed using the Nvidia Nsight Systems application, serving as the system’s native file, while the latter stores the parameters from the completed runs in a database format. Both types of files are stored within the performanceNsight directory, ensuring organized data management and easy retrieval for performance analysis.

## Source Code
To integrate the library provided by MMUOnCPU. cu into your TargetFile, please include the necessary library calls within your source code.  After incorporating the library, compile the project using the specified command below.
To complie:
`nvcc -rdc=true "TargetFile" MMUOnCPU.cu -lpthread`
