from pytorch3d.structures import Meshes, Pointclouds
import torch.nn.functional as F
import torch
from lib.common.render_utils import face_vertices
from lib.dataset.mesh_util import SMPLX, barycentric_coordinates_of_projection
from kaolin.ops.mesh import check_sign, face_normals
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from lib.dataset.Evaluator import point_mesh_distance
from lib.dataset.ECON_Evaluator import econ_point_mesh_distance


def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.norm(x - y, dim=-1) if torch.__version__ >= '1.7.0' else torch.pow(x - y, p).sum(2)**(1/p)
    
    return dist

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist=[]
        chunk=10000
        for i in range(0,x.shape[0],chunk):
            dist.append(distance_matrix(x[i:i+chunk], self.train_pts, self.p))
            
        dist = torch.cat(dist, dim=0)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels],labels

class PointFeat:

    def __init__(self, verts, faces):

        # verts [B, N_vert, 3]
        # faces [B, N_face, 3]
        # triangles [B, N_face, 3, 3]

        self.Bsize = verts.shape[0]
        self.mesh = Meshes(verts, faces)
        self.device = verts.device
        self.faces = faces

        # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
        # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
        # 2. fill mouth holes with 30 more faces

        if verts.shape[1] == 10475:
            faces = faces[:, ~SMPLX().smplx_eyeball_fid_mask]
            mouth_faces = (torch.as_tensor(
                SMPLX().smplx_mouth_fid).unsqueeze(0).repeat(
                    self.Bsize, 1, 1).to(self.device))
            self.faces = torch.cat([faces, mouth_faces], dim=1).long()

        self.verts = verts
        self.triangles = face_vertices(self.verts, self.faces)

    def get_face_normals(self):
        return face_normals(self.verts, self.faces)
    
    def get_nearest_point(self,points):
        # points [1, N, 3]
        # find nearest point on mesh

        #devices = points.device
        points=points.squeeze(0)
        nn_class=NN(X=self.verts.squeeze(0),Y=self.verts.squeeze(0),p=2)
        nearest_points,nearest_points_ind=nn_class.predict(points)
        
        # closest_triangles = torch.gather(
        #     self.triangles, 1,
        #     pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        # bary_weights = barycentric_coordinates_of_projection(
        #     points.view(-1, 3), closest_triangles)
        
        # bary_weights=F.normalize(bary_weights, p=2, dim=1)

        # normals = face_normals(self.triangles)

        # # make the lenght of the normal is 1
        # normals = F.normalize(normals, p=2, dim=2)


        # # get the normal of the closest triangle
        # closest_normals = torch.gather(
        #     normals, 1,
        #     pts_ind[:, :, None].expand(-1, -1, 3)).view(-1, 3)
        

        return nearest_points,nearest_points_ind  # on cpu

    def query_barycentirc_feats(self,points,feats):
        # feats [B,N,C]

        residues, pts_ind, _ = point_to_mesh_distance(points, self.triangles)
        closest_triangles = torch.gather(
            self.triangles, 1,
            pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(
            points.view(-1, 3), closest_triangles)

        feat_arr=feats
        feat_dim = feat_arr.shape[-1]
        feat_tri = face_vertices(feat_arr, self.faces)        
        closest_feats = torch.gather(   # query点距离最近的face的三个点的feature
                    feat_tri, 1,
                    pts_ind[:, :, None,
                            None].expand(-1, -1, 3,
                                         feat_dim)).view(-1, 3, feat_dim)
        pts_feats = ((closest_feats *
                        bary_weights[:, :, None]).sum(1).unsqueeze(0)) # 用barycentric weight加权求和
        return pts_feats.view(self.Bsize,-1,feat_dim)

    def query(self, points, feats={}):

        # points [B, N, 3]
        # feats {'feat_name': [B, N, C]}

        del_keys = ["smpl_verts", "smpl_faces", "smpl_joint","smpl_sample_id"]

        residues, pts_ind, _ = point_to_mesh_distance(points, self.triangles)
        closest_triangles = torch.gather(
            self.triangles, 1,
            pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(
            points.view(-1, 3), closest_triangles)

        out_dict = {}

        for feat_key in feats.keys():

            if feat_key in del_keys:
                continue

            elif feats[feat_key] is not None:
                feat_arr = feats[feat_key]
                feat_dim = feat_arr.shape[-1]
                feat_tri = face_vertices(feat_arr, self.faces)
                closest_feats = torch.gather(   # query点距离最近的face的三个点的feature
                    feat_tri, 1,
                    pts_ind[:, :, None,
                            None].expand(-1, -1, 3,
                                         feat_dim)).view(-1, 3, feat_dim)
                pts_feats = ((closest_feats *
                              bary_weights[:, :, None]).sum(1).unsqueeze(0)) # 用barycentric weight加权求和
                out_dict[feat_key.split("_")[1]] = pts_feats

            else:
                out_dict[feat_key.split("_")[1]] = None

        if "sdf" in out_dict.keys():
            pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))
            pts_signs = 2.0 * (
                check_sign(self.verts, self.faces[0], points).float() - 0.5)
            pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
            out_dict["sdf"] = pts_sdf

        if "vis" in out_dict.keys():
            out_dict["vis"] = out_dict["vis"].ge(1e-1).float()

        if "norm" in out_dict.keys():
            pts_norm = out_dict["norm"] * torch.tensor([-1.0, 1.0, -1.0]).to(
                self.device)
            out_dict["norm"] = F.normalize(pts_norm, dim=2)

        if "cmap" in out_dict.keys():
            out_dict["cmap"] = out_dict["cmap"].clamp_(min=0.0, max=1.0)

        for out_key in out_dict.keys():
            out_dict[out_key] = out_dict[out_key].view(
                self.Bsize, -1, out_dict[out_key].shape[-1])

        return out_dict




class ECON_PointFeat:
    def __init__(self, verts, faces):

        # verts [B, N_vert, 3]
        # faces [B, N_face, 3]
        # triangles [B, N_face, 3, 3]

        self.Bsize = verts.shape[0]
        self.device = verts.device
        self.faces = faces

        # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
        # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
        # 2. fill mouth holes with 30 more faces

        if verts.shape[1] == 10475:
            faces = faces[:, ~SMPLX().smplx_eyeball_fid_mask]
            mouth_faces = (
                torch.as_tensor(SMPLX().smplx_mouth_fid).unsqueeze(0).repeat(self.Bsize, 1,
                                                                             1).to(self.device)
            )
            self.faces = torch.cat([faces, mouth_faces], dim=1).long()

        self.verts = verts.float()
        self.triangles = face_vertices(self.verts, self.faces)
        self.mesh = Meshes(self.verts, self.faces).to(self.device)

    def query(self, points):

        points = points.float()
        residues, pts_ind = econ_point_mesh_distance(self.mesh, Pointclouds(points), weighted=False)  # 这个和ECON的不太一样

        closest_triangles = torch.gather(
            self.triangles, 1, pts_ind[None, :, None, None].expand(-1, -1, 3, 3)
        ).view(-1, 3, 3)
        bary_weights = barycentric_coordinates_of_projection(points.view(-1, 3), closest_triangles)

        feat_normals = face_vertices(self.mesh.verts_normals_padded(), self.faces)
        closest_normals = torch.gather(
            feat_normals, 1, pts_ind[None, :, None, None].expand(-1, -1, 3, 3)
        ).view(-1, 3, 3)
        shoot_verts = ((closest_triangles * bary_weights[:, :, None]).sum(1).unsqueeze(0))

        pts2shoot_normals = points - shoot_verts
        pts2shoot_normals = pts2shoot_normals / torch.norm(pts2shoot_normals, dim=-1, keepdim=True)

        shoot_normals = ((closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0))
        shoot_normals = shoot_normals / torch.norm(shoot_normals, dim=-1, keepdim=True)
        angles = (pts2shoot_normals * shoot_normals).sum(dim=-1).abs()

        return (torch.sqrt(residues).unsqueeze(0), angles)