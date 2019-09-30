import ggems

# Setting the verbosity
ggems.Verbosity(1)

# Initializing OpenCL
opencl_manager = ggems.OpenCLManager()

# Activate a context
opencl_manager.set_context_index(0)

# Printing informations about OpenCL
opencl_manager.print_info()
opencl_manager.print_device()
opencl_manager.print_build_options()
opencl_manager.print_context()
opencl_manager.print_command_queue()
opencl_manager.print_activated_context()

# Printing infos about RAM manager
opencl_manager.print_RAM()
