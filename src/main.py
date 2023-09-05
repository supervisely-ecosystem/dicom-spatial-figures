import os

from dotenv import load_dotenv

import supervisely as sly
import numpy as np

# for convenient debug, has no effect in production
if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

# check the workspace exists
workspace_id = sly.env.workspace_id()
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    sly.logger.warning("You should put correct WORKSPACE_ID value to local.env")
    raise ValueError(f"Workspace with id={workspace_id} not found")

################################    Part 1    ######################################
###################    create empty project and dataset    #########################
############################    upload volume    ###################################

dicom_dir_name = "data/CTChest_dcm"
nrrd_dir_name = "data/CTChest_nrrd"

# create empty project and dataset on server
project = api.project.create(
    workspace.id,
    name="Volumes Demo",
    type=sly.ProjectType.VOLUMES,
    change_name_if_conflict=True,
)
dataset_dcm = api.dataset.create(project.id, name="CTChest_dcm")
dataset_nrrd = api.dataset.create(project.id, name="CTChest_nrrd")
sly.logger.info(f"Project has been sucessfully created, id={project.id}")

name = "CTChest.nrrd"

# OPTION 1
# upload DICOM series to the dataset on server
series_infos = sly.volume.inspect_dicom_series(root_dir=dicom_dir_name)

for _, files in series_infos.items():
    dicom_volume_np, dicom_volume_meta = sly.volume.read_dicom_serie_volume_np(
        files, anonymize=True
    )
    progress_cb = sly.tqdm_sly(
        desc=f"Upload DICOM volume as {name}", total=sum(dicom_volume_np.shape)
    ).update
    dicom_info = api.volume.upload_np(
        dataset_dcm.id, name, dicom_volume_np, dicom_volume_meta, progress_cb
    )
    sly.logger.info(
        f"DICOM volume has been uploaded to Supervisely with ID: {dicom_info.id}"
    )

# OPTION 2
# upload NRRD volume as nparray

nrrd_path = os.path.join(nrrd_dir_name, name)
nrrd_volume_np, nrrd_volume_meta = sly.volume.read_nrrd_serie_volume_np(nrrd_path)

# use nrrd_info instead of dicom_info further down the tutorial if needed
progress_cb = sly.tqdm_sly(
    desc=f"Upload NRRD volume {name}", total=sum(nrrd_volume_np.shape)
).update
nrrd_info = api.volume.upload_np(
    dataset_id=dataset_nrrd.id,
    name=name,
    np_data=nrrd_volume_np,
    meta=nrrd_volume_meta,
    progress_cb=progress_cb,
)
sly.logger.info(f"NRRD volume has been uploaded to Supervisely with ID: {nrrd_info.id}")

# create annotation classes
lung_obj_class = sly.ObjClass("lung", sly.Mask3D)
body_obj_class = sly.ObjClass("body", sly.Bitmap)

# create project meta with all classes and upload them to server
project_meta = sly.ProjectMeta(obj_classes=[lung_obj_class, body_obj_class])
api.project.update_meta(project.id, project_meta.to_json())


################################   Part 2   #######################################
###################### Create object and Bitmap geometry ##########################
################################ Create figure ####################################
####################### Create volume slice and plane #############################

mask_dir_name = "data/mask"

# OPTION 1
# load raw geometry for 'body' as base64 encoded string
raw_bitmap_data = "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"
# convert 'body' raw data to ndarray
bitmap_data_decoded = sly.Bitmap.base64_2_data(raw_bitmap_data)

# OPTION 2
# load ndarray from npy file
body_path = os.path.join(mask_dir_name, "body.npy")
bitmap_data = np.load(body_path)

# create VolumeObject
body = sly.VolumeObject(body_obj_class)

# PointLocation is a top-left corner of the mask location for 'body' geometry
# use if the mask size is smaller than the image size othervise leave as 'None'
bitmap_origin = sly.PointLocation(91, 36)

# create geometry for 'body'
bitmap_geometry = sly.Bitmap(bitmap_data, bitmap_origin)

# create figure for 'body' with loaded Bitmap geometry
bitmap_figure = sly.VolumeFigure(body, bitmap_geometry, "axial", 69)

# add figure to volume slice
volume_slice = sly.Slice(69, figures=[bitmap_figure])

# connect the slice to the corresponding plane
plane = sly.Plane(plane_name="axial", items=[volume_slice], volume_meta=dicom_info.meta)


################################    Part 3    #####################################
###################### Create object and load Mask3D geometry ##########################
############################## Create empty figure ################################

mask_dir_name = "data/mask"

# path to the generated 'lung' geometry, represented as a .nrrd file
mask_path = os.path.join(mask_dir_name, "lung.nrrd")

# load geometry for 'lung' as bytes
with open(mask_path, "rb") as file:
    geometry_bytes = file.read()

# create VolumeObject
lung = sly.VolumeObject(lung_obj_class)

# create figure for 'lung' with empty Mask3D geometry
empty_figure = sly.VolumeFigure(
    lung, sly.Mask3D(np.zeros((3, 3, 3), dtype=np.bool_)), None, None
)

################################    Part 4    ######################################
###################  Create object collection, volume annotation  ##################
################### and upload annotation to the volume on server ##################

# create VolumeObjectCollection
objects = sly.VolumeObjectCollection([lung, body])

# create result volume annotation with the figure on the corresponding plane
# and upload an empty spatial figure
volume_ann = sly.VolumeAnnotation(
    dicom_info.meta,
    objects,
    plane_axial=plane,
    spatial_figures=[empty_figure],
)

key_id_map = sly.KeyIdMap()

# upload annotation to the volume on server
api.volume.annotation.append(dicom_info.id, volume_ann, key_id_map)
sly.logger.info(
    f"Annotation has been sucessfully uploaded to the volume {dicom_info.name} in dataset with ID={dicom_info.dataset_id}"
)

# upload prepared 'lung' geometry into its figure
api.volume.figure.upload_sf_geometry(
    volume_ann.spatial_figures, [geometry_bytes], key_id_map
)
sp_figure_id = key_id_map.get_figure_id(volume_ann.spatial_figures[0].key())
sly.logger.info(
    f"Geometry has been sucessfully uploaded to the spatial figure with ID={sp_figure_id}"
)
