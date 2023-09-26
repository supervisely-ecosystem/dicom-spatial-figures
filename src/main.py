import os
import numpy as np
import cv2
from dotenv import load_dotenv
import supervisely as sly

# To init API for communicating with Supervisely Instance.
# It needs to load environment variables with credentials and workspace ID
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

mask3d_path = "data/mask/lung.nrrd"
image_path = "data/mask/body.png"

# create annotation classes
lung_class = sly.ObjClass("lung", sly.Mask3D, color=[111, 107, 151])
body_class = sly.ObjClass("body", sly.Mask3D, color=[209, 192, 129])
fbody_class = sly.ObjClass("foreign_body", sly.Mask3D, color=[255, 153, 204])

# update project meta with new classes
api.project.append_classes(project_info.id, [lung_class, fbody_class, body_class])

# create Mask3D object for 'lung' annotation using NRRD file with 3D Object
lung_mask = sly.Mask3D.from_file(mask3d_path)
lung = sly.VolumeObject(lung_class, mask_3d=lung_mask)


# create Mask3D object for 'foreign_body' annotation using NumPy array representing the sphere
width, height, depth = (512, 512, 139)  # volume shape
center = np.array([128, 242, 69])  # sphere center in the volume
radius = 25
x, y, z = np.ogrid[:width, :height, :depth]
fbody_mask = (
    (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= radius**2
).astype(np.uint8)
fbody_mask = sly.Mask3D(fbody_mask)
foreign_body = sly.VolumeObject(fbody_class, mask_3d=fbody_mask)


# create Mask3D object for 'body' annotation using image file
mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# create an empty mask with the same dimensions as the volume
body_mask = sly.Mask3D(np.zeros(volume_info.file_meta["sizes"], np.bool_))
# fill this mask with the an image mask for the desired plane
body_mask.add_mask_2d(mask, plane_name="axial", slice_index=69, origin=[36, 91])
body = sly.VolumeObject(body_class, mask_3d=body_mask)

################################    Part 3    ######################################
################### Upload annotation into the volume on server ####################
####################################################################################

# create volume annotation object
volume_ann = sly.VolumeAnnotation(
    volume_info.meta,
    objects=[lung, foreign_body, body],
    spatial_figures=[lung.figure, foreign_body.figure, body.figure],
)

# upload VolumeAnnotation
api.volume.annotation.append(volume_info.id, volume_ann)
sly.logger.info(
    f"Annotation has been sucessfully uploaded to the volume {volume_info.name} in dataset with ID={volume_info.dataset_id}"
)
