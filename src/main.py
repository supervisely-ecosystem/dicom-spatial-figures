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

# create annotation classes
lung_class = sly.ObjClass("lung", sly.Mask3D, color=[111, 107, 151])
body_class = sly.ObjClass("body", sly.Mask3D, color=[209, 192, 129])
tumor_class = sly.ObjClass("tumor", sly.Mask3D, color=[255, 153, 204])

# update project meta with new classes
api.project.append_classes(project_info.id, [lung_class, tumor_class, body_class])

################################  1  NRRD file    ######################################

mask3d_path = "data/mask/lung.nrrd"

# create 3D Mask annotation for 'lung' using NRRD file with 3D object
lung_mask = sly.Mask3D.create_from_file(mask3d_path)
lung = sly.VolumeObject(lung_class, mask_3d=lung_mask)

###############################  2  NumPy array    #####################################


def generate_tumor_array():
    """
    Generate a NumPy array representing the tumor as a sphere
    """
    width, height, depth = (512, 512, 139)  # volume shape
    center = np.array([128, 242, 69])  # sphere center in the volume
    radius = 25
    x, y, z = np.ogrid[:width, :height, :depth]
    # Calculate the squared distances from each point to the center
    squared_distances = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    # Create a boolean mask by checking if squared distances are less than or equal to the square of the radius
    tumor_array = squared_distances <= radius**2
    tumor_array = tumor_array.astype(np.uint8)
    return tumor_array


# create 3D Mask annotation for 'tumor' using NumPy array
tumor_mask = sly.Mask3D(generate_tumor_array())
tumor = sly.VolumeObject(tumor_class, mask_3d=tumor_mask)

##################################  3  Image    ########################################

image_path = "data/mask/body.png"

# create 3D Mask annotation for 'body' using image file
mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# create an empty mask with the same dimensions as the volume
body_mask = sly.Mask3D(np.zeros(volume_info.file_meta["sizes"], np.bool_))
# fill this mask with the an image mask for the desired plane.
# to avoid errors, use constants: Plane.AXIAL, Plane.CORONAL, Plane.SAGITTAL
body_mask.add_mask_2d(mask, plane_name=sly.Plane.AXIAL, slice_index=69, origin=[36, 91])
body = sly.VolumeObject(body_class, mask_3d=body_mask)

# create volume annotation object
volume_ann = sly.VolumeAnnotation(
    volume_info.meta,
    objects=[lung, tumor, body],
    spatial_figures=[lung.figure, tumor.figure, body.figure],
)

# upload VolumeAnnotation
api.volume.annotation.append(volume_info.id, volume_ann)
sly.logger.info(
    f"Annotation has been sucessfully uploaded to the volume {volume_info.name} in dataset with ID={volume_info.dataset_id}"
)
