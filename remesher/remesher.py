import torch

from graph import Mesh, NodeType, interpolate_field, cells_to_edges, find_edge, find_edges_with
from utils import get_triangle_aspect_ratio, get_triangle_sarea
from remesher.core import MeshState, EPS, AR_THRESHOLD, closest_SDP, to_meshstate, from_meshstate

from typing import Any

#! THIS IS A WORK IN PROGRESS...
#! Extremyly inefficient implementation, needs to be optimized
#! Try not to use this class if possible (but no alternative for now)
class Remesher:
    def __init__(self) -> None:
        pass

    def __is_border(self, ms: MeshState, edge_id) -> bool:
        return (ms.opposites[edge_id][-1] == -1).item()

    def __get_splittable_edges_mask(self, ms: MeshState) -> torch.Tensor:
        """ Find splittable edges """
        S_ij = 0.5 * (ms.S_i[ms.edges[:, 0]] + ms.S_i[ms.edges[:, 1]])
        u_ij = ms.mesh_pos[ms.edges[:, 1]] - ms.mesh_pos[ms.edges[:, 0]]
        edge_size = torch.sqrt(torch.einsum("ei,eij,ej->e", u_ij, S_ij, u_ij))
        return (edge_size > 1 + EPS)

    def __get_flippable_edges_mask(self, ms: MeshState) -> torch.Tensor:
        """ Find flippable edges """
        i, j = ms.edges[:, 0], ms.edges[:, 1]
        k, l = ms.opposites[:, 0], ms.opposites[:, 1]

        border_mask = (l == -1)

        u_ik = ms.mesh_pos[i] - ms.mesh_pos[k]
        u_jk = ms.mesh_pos[j] - ms.mesh_pos[k]
        u_il = ms.mesh_pos[i] - ms.mesh_pos[l]
        u_jl = ms.mesh_pos[j] - ms.mesh_pos[l]
        S_A = 0.5 * (ms.S_i[i] + ms.S_i[j] + ms.S_i[k] + ms.S_i[l])

        ujk_x_uik = u_jk[:, 0] * u_ik[:, 1] - u_jk[:, 1] * u_ik[:, 0]
        uil_x_ujl = u_il[:, 0] * u_jl[:, 1] - u_il[:, 1] * u_jl[:, 0]
        uil_SA_ujl = torch.einsum("ei,eij,ej->e", u_il, S_A, u_jl)
        ujk_SA_uik = torch.einsum("ei,eij,ej->e", u_jk, S_A, u_ik)
        flippable = ujk_x_uik*uil_SA_ujl + ujk_SA_uik*uil_x_ujl < EPS
        flippable &= ~border_mask

        return flippable

    def __get_maximal_independent_set(self, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Find a maximal independent set """
        maximal_independent_set = []
        node_set = set()
        for e_id in torch.nonzero(mask):
            if edges[e_id,0].item() not in node_set and edges[e_id, 1].item() not in node_set:
                maximal_independent_set.append(e_id)
                node_set.add(edges[e_id, 0].item())
                node_set.add(edges[e_id, 1].item())

        return torch.Tensor(maximal_independent_set).long()

    # TODO: add a stopping contidition (max_iter ?)
    def __flip(self, ms: MeshState, new_edge_mask: torch.Tensor) -> MeshState:
        edge_mask = new_edge_mask
        edge_mask &= ms.node_mask[ms.edges[:, 0]] | ms.node_mask[ms.edges[:, 1]]
        while True:
            flippable_mask = self.__get_flippable_edges_mask(ms) & edge_mask
            maximal_independent_set = self.__get_maximal_independent_set(ms.edges, flippable_mask)
            if len(maximal_independent_set) == 0: break

            def correct_opposites(nodes: tuple[Any,Any], old, new) -> None:
                e_id = find_edge(nodes[0], nodes[1], updated_edges)
                updated_opposites[e_id][updated_opposites[e_id] == old] = new

            updated_edges = ms.edges.clone()
            updated_opposites = ms.opposites.clone()
            for edge_id in maximal_independent_set:
                i, j = updated_edges[edge_id]
                k, l = updated_opposites[edge_id]

                if find_edge(k, l, updated_edges) != -1: continue
                if get_triangle_aspect_ratio(ms.mesh_pos, i, k, l) > AR_THRESHOLD: continue
                if get_triangle_aspect_ratio(ms.mesh_pos, j, k, l) > AR_THRESHOLD: continue

                # check if a triangle would be inverted
                tris = torch.stack([
                    torch.stack([ms.mesh_pos[i], ms.mesh_pos[k], ms.mesh_pos[j]], dim=0),
                    torch.stack([ms.mesh_pos[i], ms.mesh_pos[k], ms.mesh_pos[l]], dim=0),
                    torch.stack([ms.mesh_pos[j], ms.mesh_pos[i], ms.mesh_pos[k]], dim=0),
                    torch.stack([ms.mesh_pos[j], ms.mesh_pos[l], ms.mesh_pos[k]], dim=0)
                ], dim=0)
                tris_orientation = torch.sign(get_triangle_sarea(tris))
                if tris_orientation[0] != tris_orientation[1] or tris_orientation[2] != tris_orientation[3]: continue

                correct_opposites((i, k), j, l)
                correct_opposites((i, l), j, k)
                correct_opposites((j, k), i, l)
                correct_opposites((j, l), i, k)

                updated_edges[edge_id]     = torch.Tensor([k, l])
                updated_opposites[edge_id] = torch.Tensor([i, j])

            edge_mask[maximal_independent_set] = False  # edges checked

            ms = ms._replace(
                edges     = updated_edges,
                opposites = updated_opposites,
            )

        return ms

    # TODO: add a stopping contidition (max_iter ?)
    def __split(self, ms: MeshState) -> MeshState:
        """ Split all possible edges in `ms` """

        def add_edge(nodes: tuple[Any,Any], opposites: tuple[Any,Any]) -> None:
            new_edges_buf.append(nodes)
            new_opposites_buf.append(opposites)

        def correct_opposites(nodes: tuple[Any,Any], old, new) -> None:
            e_id = find_edge(nodes[0], nodes[1], updated_edges)
            updated_opposites[e_id][updated_opposites[e_id] == old] = new
            # updated_edge_mask[e_id] = True

        def split_edge(i, j, k, l, m) -> bool:
            # check if new triangles would have a bad aspect ratio
            if get_triangle_aspect_ratio(updated_mesh_pos, i, m, k) > AR_THRESHOLD: return False
            if get_triangle_aspect_ratio(updated_mesh_pos, j, m, k) > AR_THRESHOLD: return False
            if l != -1:
                if get_triangle_aspect_ratio(updated_mesh_pos, i, m, l) > AR_THRESHOLD: return False
                if get_triangle_aspect_ratio(updated_mesh_pos, j, m, l) > AR_THRESHOLD: return False

            # add new edges and their corresponding opposite nodes
            add_edge((i, m), (k, l))
            add_edge((j, m), (k, l))
            add_edge((k, m), (i, j))
            if l != -1:
                add_edge((l, m), (i, j))

            # correct opposite nodes of adjacent edges
            correct_opposites((i, k), j, m)
            correct_opposites((j, k), i, m)
            if l != -1:
                correct_opposites((i, l), j, m)
                correct_opposites((j, l), i, m)

            return True

        edge_mask = ms.node_mask[ms.edges[:, 0]] | ms.node_mask[ms.edges[:, 1]]
        modified_edge_mask = torch.zeros(len(ms.edges)).bool()
        updated_edge_mask = torch.zeros(len(ms.edges)).bool()   # mask used by the flip function
        while True:
            invalid_edge_mask = self.__get_splittable_edges_mask(ms) & edge_mask
            maximal_independent_set = self.__get_maximal_independent_set(ms.edges, invalid_edge_mask)
            if len(maximal_independent_set) == 0: break

            # interpolate values
            i_, j_ = ms.edges[maximal_independent_set][:,0], ms.edges[maximal_independent_set][:,1]
            m_mesh_pos  = 0.5 * (ms.mesh_pos[i_] + ms.mesh_pos[j_])
            m_world_pos = 0.5 * (ms.world_pos[i_] + ms.world_pos[j_])
            m_S_i       = 0.5 * (ms.S_i[i_] + ms.S_i[j_])
            m_S_i       = torch.stack([closest_SDP(S) for S in torch.unbind(m_S_i, dim=0)], dim=0)
            updated_node_mask = torch.concat([ms.node_mask.clone(), torch.Tensor([True]*len(maximal_independent_set))])
            updated_mesh_pos  = torch.concat([ms.mesh_pos.clone(),  m_mesh_pos])
            updated_world_pos = torch.concat([ms.world_pos.clone(), m_world_pos])
            updated_S_i       = torch.concat([ms.S_i.clone(),       m_S_i])

            # split edges
            updated_edges = ms.edges.clone()
            updated_opposites = ms.opposites.clone()
            new_edges_buf = []
            new_opposites_buf = []
            removed_edges, kept_edges = [], []
            for it, edge_id in enumerate(maximal_independent_set):
                i, j = i_[it], j_[it]
                k, l = updated_opposites[edge_id].unbind()
                m = len(ms.node_mask) + it
                if split_edge(i, j, k, l, m):
                    removed_edges.append(edge_id)
                else:
                    kept_edges.append(edge_id)

            removed_edges = torch.Tensor(removed_edges).long()
            kept_edges    = torch.Tensor(kept_edges).long()

            new_edges     = torch.Tensor(new_edges_buf)
            new_opposites = torch.Tensor(new_opposites_buf)
            
            removed_edges_mask = torch.zeros(len(updated_edges)).bool()
            removed_edges_mask[removed_edges] = True

            updated_edges = updated_edges[~removed_edges_mask]
            updated_edges = torch.concat([updated_edges, new_edges]).long()
            updated_opposites = updated_opposites[~removed_edges_mask]
            updated_opposites = torch.concat([updated_opposites, new_opposites]).long()
            modified_edge_mask = modified_edge_mask[~removed_edges_mask]
            modified_edge_mask = torch.concat([modified_edge_mask, torch.ones(len(new_edges)).bool()])

            edge_mask[kept_edges] = False
            edge_mask = edge_mask[~removed_edges_mask]
            edge_mask = torch.concat([edge_mask, torch.ones(len(new_edges))]).bool()

            ms = MeshState(
                edges = updated_edges.long(),
                opposites = updated_opposites.long(),
                modified_edge_mask = modified_edge_mask,
                node_mask = updated_node_mask.bool(),
                mesh_pos = updated_mesh_pos,
                world_pos = updated_world_pos,
                S_i = updated_S_i
            )

            updated_edge_mask = updated_edge_mask[~removed_edges_mask]
            if len(new_edges) > 0:
                updated_edge_mask = torch.concat([updated_edge_mask, torch.ones(len(new_edges)).bool()])
            ms = self.__flip(ms, updated_edge_mask)
            
        return ms

    # TODO: group calls the find_edge (and other functions if possible)
    def __collapse(self, ms: MeshState) -> MeshState:
        """ Collapse all possible edges in `ms` """
        h: float = 0.01  # "close to invalid" constant

        def check_face(prev_tri: list, new_tri: list, prev_edges: list, new_edges: list[list]) -> bool:
            # check if the new face would change the border
            boundary_condition = not self.__is_border(ms, prev_edges[0]) and not self.__is_border(ms, prev_edges[1])
            if not boundary_condition: return False

            # check if the new face would have a bad aspect ratio
            aspect_ratio = get_triangle_aspect_ratio(ms.mesh_pos, *new_tri).item()
            if aspect_ratio >= AR_THRESHOLD: return False

            # check if the new face would be inverted
            tris = torch.stack([ms.mesh_pos[prev_tri], ms.mesh_pos[new_tri]], dim=0)
            tris_orientation = torch.sign(get_triangle_sarea(tris))
            if tris_orientation[0].item() != tris_orientation[1].item(): return False

            # FIXME: this removes almost every edges
            # check if the new face's edges would be close to invalid
            edges = tris = torch.stack([
                torch.Tensor(new_edges[0]),
                torch.Tensor(new_edges[1])
            ], dim=0).long()
            S_ij = 0.5 * (ms.S_i[edges[:, 0]] + ms.S_i[edges[:, 1]])
            u_ij = ms.mesh_pos[edges[:, 1]] - ms.mesh_pos[edges[:, 0]]
            edge_size = torch.sqrt(torch.einsum("ei,eij,ej->e", u_ij, S_ij, u_ij))
            if bool(torch.any(edge_size > 1 - h).item()): return False

            return True
        
        def check_collapisble(i, j, k, l) -> bool:
            ij = find_edge(i, j, ms.edges)
            
            # find all edges from `j` (without `i,j`)
            edges_j = find_edges_with(j, ms.edges)
            edges_j = edges_j[edges_j != ij]

            # find all nodes that are at the end of the edges found previously
            adjacent_nodes = ms.edges[edges_j].flatten()
            adjacent_nodes = adjacent_nodes[adjacent_nodes != i]
            adjacent_nodes = adjacent_nodes[adjacent_nodes != j]
            
            # check all edges for "forbidden" cases
            checked_tris = set()
            for e_id in edges_j:
                nodes = ms.edges[e_id]
        
                adjacent_id = nodes[nodes != j]
                for node_id in adjacent_nodes[adjacent_nodes != adjacent_id]:
                    adjacent_edge = find_edge(adjacent_id, node_id, ms.edges)
                    if adjacent_edge == -1: continue

                    tri, order = torch.sort(torch.Tensor([j, node_id, adjacent_id]))
                    tri = tri.long().tolist()

                    if tuple(tri) not in checked_tris:
                        checked_tris.add(tuple(tri))

                        new_tri = torch.Tensor([i, node_id, adjacent_id])[order].long().tolist()
                        if not check_face(tri, new_tri, [e_id, adjacent_edge], [[i, node_id], [i, adjacent_id]]):
                            return False

            return True

        deleted_nodes = []
        while torch.count_nonzero(ms.modified_edge_mask) > 0:
            edge_id = torch.nonzero(ms.modified_edge_mask)[0,0]
            ms.modified_edge_mask[edge_id] = False

            nodes = [*ms.edges[edge_id], *ms.opposites[edge_id]]
            collapsible = check_collapisble(*nodes)
            if not collapsible:
                nodes[0], nodes[1] = nodes[1], nodes[0] # swap `i` and`j`
                collapsible = check_collapisble(*nodes)

            if collapsible:
                deleted_edges_buf = []
                added_edges_buf = []
                added_opposites_buf = []    # TODO
                updated_edge_mask = torch.zeros(len(ms.edges)).bool()   # mask used by the flip function
                
                ji = find_edge(nodes[1], nodes[0], ms.edges)
                jk = find_edge(nodes[1], nodes[2], ms.edges)
                jl = find_edge(nodes[1], nodes[3], ms.edges)

                edges_j = find_edges_with(nodes[1], ms.edges)
                edges_j = edges_j[(edges_j != jk) & (edges_j != jl) & (edges_j != ji)]

                adjacent_nodes = ms.edges[edges_j].flatten()
                adjacent_nodes = adjacent_nodes[adjacent_nodes != nodes[0]]
                adjacent_nodes = adjacent_nodes[adjacent_nodes != nodes[1]]
                adjacent_nodes = torch.concat([adjacent_nodes, torch.Tensor([nodes[2], nodes[3]]).long()])

                deleted_edges_buf.append(ji)
                deleted_edges_buf.append(jk)
                deleted_edges_buf.append(jl)
                deleted_nodes.append(nodes[1].item())
                for e in edges_j:
                    deleted_edges_buf.append(e)
        
                    adjacent_node = ms.edges[e][ms.edges[e] != nodes[1]]
                    added_edges_buf.append([nodes[0], adjacent_node])
                    added_opposites_buf.append([ms.opposites[e][0], ms.opposites[e][1]])

                    for node_id in adjacent_nodes[adjacent_nodes != adjacent_node]:
                        adjacent_edge = find_edge(adjacent_node, node_id, ms.edges)
                        if adjacent_edge == -1: continue

                        # update opposite nodes of adjacent edges
                        updated_edge_mask[adjacent_edge] = True
                        ms.opposites[adjacent_edge][ms.opposites[adjacent_edge] == nodes[1]] = nodes[0]

                deleted_edge_mask = torch.ones(len(ms.edges)).bool()
                deleted_edge_mask[torch.Tensor(deleted_edges_buf).long()] = False

                updated_modified_edge_mask = torch.concat([ms.modified_edge_mask[deleted_edge_mask], torch.zeros(len(added_edges_buf)).bool()])
                updated_edges = torch.concat([ms.edges[deleted_edge_mask], torch.Tensor(added_edges_buf).long()])
                updated_opposites = torch.concat([ms.opposites[deleted_edge_mask], torch.Tensor(added_opposites_buf).long()])

                updated_edge_mask = updated_edge_mask[deleted_edge_mask]
                if len(added_edges_buf) > 0:
                    updated_edge_mask = torch.concat([updated_edge_mask, torch.ones(len(added_edges_buf)).bool()])

                ms = ms._replace(
                    edges = updated_edges,
                    opposites = updated_opposites,
                    modified_edge_mask = updated_modified_edge_mask
                )

                ms = self.__flip(ms, updated_edge_mask)
        
        # remove deleted nodes and reindex other nodes
        deleted_node_mask = torch.zeros(len(ms.node_mask)).bool()
        deleted_node_mask[deleted_nodes] = True

        index_map = torch.cumsum(~deleted_node_mask, dim=0) - 1
        ms = ms._replace(
            edges = index_map[ms.edges],
            opposites = index_map[ms.opposites],
            modified_edge_mask = torch.zeros(len(ms.edges)).bool(),
            node_mask = ms.node_mask[~deleted_node_mask],
            mesh_pos = ms.mesh_pos[~deleted_node_mask],
            world_pos = ms.world_pos[~deleted_node_mask],
            S_i = ms.S_i[~deleted_node_mask]
        )

        return ms

    def __call__(self, mesh: Mesh, sizing_field: torch.Tensor) -> Mesh:
        ms = to_meshstate(mesh, sizing_field)
        ms = self.__split(ms)
        ms = self.__collapse(ms)
        return mesh
        # return from_meshstate(ms)
