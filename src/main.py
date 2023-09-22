import os
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import supervisely as sly

# for convenient debug, has no effect in production
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

# check the workspace exists
workspace_id = sly.env.workspace_id()
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    sly.logger.warning("You should put correct WORKSPACE_ID value to local.env")
    raise ValueError(f"Workspace with id={workspace_id} not found")

################################    Part 1    ######################################
###################    Create empty project and dataset    #########################
############################    Upload volume    ###################################

# create empty project and dataset on server
project_info = api.project.create(
    workspace.id,
    name="Volumes Demo",
    type=sly.ProjectType.VOLUMES,
    change_name_if_conflict=True,
)
dataset_info = api.dataset.create(project_info.id, name="CTChest")

sly.logger.info(
    f"Project with id={project_info.id} and dataset with id={dataset_info.id} have been successfully created"
)

# upload NRRD volume as ndarray into dataset
volume_info = api.volume.upload_nrrd_serie_path(
    dataset_info.id,
    name="CTChest.nrrd",
    path="data/CTChest_nrrd/CTChest.nrrd",
)


################################    Part 2    #####################################
######################## Create annotations for volume ############################
###################################################################################

mask2d_path = "data/mask/body.png"
mask3d_path = "data/mask/lung.nrrd"

# create annotation classes
lung_class = sly.ObjClass("lung", sly.Mask3D, color=[111, 107, 151])
body_class = sly.ObjClass("body", sly.Mask3D, color=[209, 192, 129])

# update project meta with new classes
api.project.append_classes(project_info.id, [lung_class, body_class])

# create Mask3D object for 'body' annotation using 2D mask (image)
mask_2d = Image.open(mask2d_path)
mask_2d = np.array(mask_2d)
body_mask_3d = sly.Mask3D(np.zeros(volume_info.file_meta["sizes"], np.bool_))
body_mask_3d.add_mask_2d(mask_2d, "axial", 69, [36, 91])

# create Mask3D object for 'lung' annotation using NRRD file with 3D Object
lung_mask_3d = sly.Mask3D.from_file(mask3d_path)

# create VolumeObjects with Mask3D
lung = sly.VolumeObject(lung_class, mask_3d=body_mask_3d)
body = sly.VolumeObject(body_class, mask_3d=lung_mask_3d)

################################    Part 4    ######################################
################### Upload annotation into the volume on server ####################
####################################################################################

# create volume annotation object
volume_ann = sly.VolumeAnnotation(
    volume_info.meta,
    objects=[lung, body],
    spatial_figures=[lung.figure, body.figure],
)

# upload VolumeAnnotation
api.volume.annotation.append(volume_info.id, volume_ann)
sly.logger.info(
    f"Annotation has been sucessfully uploaded to the volume {volume_info.name} in dataset with ID={volume_info.dataset_id}"
)
