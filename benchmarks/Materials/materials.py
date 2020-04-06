from ggems import *

# ------------------------------------------------------------------------------
# STEP 1: Choosing an OpenCL context
opencl_manager.set_context_index(2)

# ------------------------------------------------------------------------------
# STEP 2: Setting GGEMS materials
materials_manager.set_materials(b"data/materials.txt")

# ------------------------------------------------------------------------------
# STEP 3: Add 1 or more material to get properties
materials = GGEMSMaterials()

materials.add_material(b"Blood")
materials.add_material(b"Lung")

materials.set_cut(b"gamma", 1.0, b"mm")
materials.set_cut(b"e+", 1.0, b"mm")
materials.set_cut(b"e-", 1.0, b"mm")

materials.initialize()

materials.print_material_properties()

# ------------------------------------------------------------------------------
# STEP 4: Exit GGEMS safely
opencl_manager.clean()
exit()
