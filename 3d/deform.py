import torch
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import open3d as o3d

import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
device = "cpu" # error when using 'mps' device

def create_triangle_mesh_o3d(verts, triangles):
    """
    Args:
        verts (np.ndarray): (V, 3) array of vertex positions
        triangles (np.ndarray): (T, 3) array of indices

    Returns:
        mesh (o3d.geometry.TriangleMesh): Open3D triangle mesh
    """
    verts_u = o3d.utility.Vector3dVector(verts)
    triangles_u = o3d.utility.Vector3iVector(triangles)
    mesh = o3d.geometry.TriangleMesh(verts_u, triangles_u)
    mesh.compute_vertex_normals()
    return mesh

print(f"Using device: {device}")

# Open obj file
verts, faces, aux = load_obj("dolphin.obj")

faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0). 
# (scale, center) will be used to bring the predicted mesh to its original center and scale
# Note that normalizing the target mesh, speeds up the optimization but is not necessary!
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale

# We construct a Meshes structure for the target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

# We initialize the source shape to be a sphere of radius 1
src_mesh = ico_sphere(4, device)

# Display source and target meshes with open3d
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
trg_mesh_o3d = create_triangle_mesh_o3d(trg_mesh.verts_packed().numpy(), trg_mesh.faces_packed().numpy())
trg_mesh_o3d.paint_uniform_color([1, 0.706, 0])

src_mesh_o3d = create_triangle_mesh_o3d(src_mesh.verts_packed().numpy(), src_mesh.faces_packed().numpy())

vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.set_full_screen(True)
vis.add_geometry(trg_mesh_o3d.translate([2.0, 0.0, 0.0]))
vis.add_geometry(src_mesh_o3d)
ctr = vis.get_view_control()
ctr.set_zoom(0.5)
ctr.rotate(-500, 150)
# vis.run()

save_image = True

# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in src_mesh
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# The optimizer
# optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
optimizer = torch.optim.AdamW([deform_verts], lr=1e-2)

# Number of optimization steps
Niter = 2000
# Weight for the chamfer loss
w_chamfer = 1.0 
# Weight for mesh edge loss
w_edge = 1.0 
# Weight for mesh normal consistency
w_normal = 0.01 
# Weight for mesh laplacian smoothing
w_laplacian = 0.1 
# Plot period for the losses
# plot_period = 50
loop = tqdm(range(Niter))
# loop = range(Niter)

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

for i in loop:
    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)

     # Visualize
    src_mesh_o3d.vertices = o3d.utility.Vector3dVector(new_src_mesh.verts_packed().detach().cpu().numpy())
    vis.update_geometry(src_mesh_o3d)
    vis.poll_events()
    vis.update_renderer()
    if save_image:
        # vis.capture_screen_image("deform_anim/temp_%04d.jpg" % i)
        vis.capture_screen_image("deform_adamw_anim/temp_%04d.jpg" % i)

    # We sample 5k points from the surface of each mesh 
    sample_trg = sample_points_from_meshes(trg_mesh, 5000)
    sample_src = sample_points_from_meshes(new_src_mesh, 5000)

    # Initialize optimizer
    optimizer.zero_grad()
    
    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)
    
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)
    
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    
    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
    
    # Print the losses
    loop.set_description('total_loss = %.6f' % loss)
    
    # Save the losses for plotting
    chamfer_losses.append(float(loss_chamfer.detach().cpu()))
    edge_losses.append(float(loss_edge.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        
    # Optimization step
    loss.backward()
    optimizer.step()

vis.destroy_window()
# # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)