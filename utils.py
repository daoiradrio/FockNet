import numpy as np

from e3x.so3.irreps import clebsch_gordan_for_degrees, clebsch_gordan
from CompChemUtils.chemdata import NAO_minimal_valence

from e3x.so3._normalization import normalization_constant
from e3x.config import Config



# decompose tensor product in to irreducible of degree l3
# approach is rather general, but here the last dimension of tensors x, y is assumend to be number of MOs
def reduce_mo_tensor_product(x, y, l1, l2, l3):
    xy = np.einsum("ik,jk->ijk", x, y)#.sum(2)
    xy = np.expand_dims(xy, axis=2)
    return normalization_constant(Config.normalization, l3) * (clebsch_gordan_for_degrees(l1, l2, l3) * xy).sum(1).sum(0)



# "invert" decomposition of tensor product (of tensors of degree l1, l2) into irreducible of degree l3
def dereduce_mo_tensor_product(x, l1, l2, l3):
    return ((x * clebsch_gordan_for_degrees(l1, l2, l3)) / normalization_constant(Config.normalization, l3)).sum(2)



def benchmark_atom_features(H, elems):
    CG = clebsch_gordan(1, 1, 2)
    CG000 = clebsch_gordan_for_degrees(0, 0, 0)
    CG11 = CG[1:, 1:, :]
    CG011 = clebsch_gordan_for_degrees(0, 1, 1)

    eps, C = np.linalg.eig(H)
    sort_idx = np.argsort(eps)[::-1]
    C = C[:, sort_idx]
    eps = eps[sort_idx]
    C_eps = np.dot(C, np.diag(eps))
    num_mos = C.shape[1]

    ss_atom_features = []
    sp_atom_features = []
    pp_atom_features = []

    iao1 = 0

    for elem in elems:

        ss_block = np.zeros((1, 1, 1))
        for imo in range(num_mos):
            ss_block[0, 0, 0] += (CG000[0, 0, 0] * C[iao1, imo] * C_eps[iao1, imo])
        ss_atom_features.append(ss_block)

        if elem != "H":
            
            pp_block = np.zeros((1, 9, 1))
            for m3 in range(9):
                for m1 in range(3):
                    for m2 in range(3):
                        for imo in range(num_mos):
                            pp_block[0, m3, 0] += (CG11[m1, m2, m3] * C[iao1+1+m1, imo] * C_eps[iao1+1+m2, imo])
            pp_atom_features.append(pp_block)
            
            sp_block = np.zeros((1, 3, 1))
            for m3 in range(3):
                for m2 in range(3):
                    for imo in range(num_mos):
                        sp_block[0, m3, 0] += (CG011[0, m2, m3] * C[iao1, imo] * C_eps[iao1+1+m2, imo])
            sp_atom_features.append(sp_block)
        
        iao1 += NAO_minimal_valence[elem]
    
    ss_atom_features = np.stack(ss_atom_features)
    sp_atom_features = np.stack(sp_atom_features)
    pp_atom_features = np.stack(pp_atom_features)

    return ss_atom_features, sp_atom_features, pp_atom_features



def benchmark_pair_features(H, elems):
    CG11 = clebsch_gordan(1, 1, 2)[1:, 1:, :]
    CG000 = clebsch_gordan_for_degrees(0, 0, 0)
    CG011 = clebsch_gordan_for_degrees(0, 1, 1)
    CG101 = clebsch_gordan_for_degrees(1, 0, 1)

    eps, C = np.linalg.eig(H)
    sort_idx = np.argsort(eps)[::-1]
    C = C[:, sort_idx]
    eps = eps[sort_idx]
    C_eps = np.dot(C, np.diag(eps))
    num_mos = C.shape[1]

    ss_pair_features = []
    sp_pair_features = []
    ps_pair_features = []
    pp_pair_features = []

    iao1 = 0

    for ielem1, elem1 in enumerate(elems):
        
        iao2 = 0

        for elem2 in elems[:ielem1]:

            ss_block = np.zeros((1, 1, 1))   
            for i in range(num_mos):
                ss_block[0, 0, 0] += (CG000[0, 0, 0] * C[iao1, i] * C_eps[iao2, i])
            ss_pair_features.append(ss_block)

            if elem1 != "H" and elem2 != "H":
                    
                sp_block = np.zeros((1, 3, 1))
                ps_block = np.zeros((1, 3, 1))
                pp_block = np.zeros((1, 9, 1))

                for m3 in range(3):
                    for m1 in range(3):
                        for imo in range(num_mos):
                            ps_block[0, m3, 0] += (CG101[m1, 0, m3] * C[iao1+1+m1, imo] * C_eps[iao2, imo])
                    for m2 in range(3):
                        for imo in range(num_mos):
                            sp_block[0, m3, 0] += (CG011[0, m2, m3] * C[iao1, imo] * C_eps[iao2+1+m2, imo])
                sp_pair_features.append(sp_block)
                ps_pair_features.append(ps_block)
                        
                for m3 in range(9):
                    for m1 in range(3):
                        for m2 in range(3):
                            for imo in range(num_mos):
                                pp_block[0, m3, 0] += (CG11[m1, m2, m3] * C[iao1+1+m1, imo] * C_eps[iao2+1+m2, imo])
                pp_pair_features.append(pp_block)

            elif elem1 == "H" and elem2 != "H":

                sp_block = np.zeros((1, 3, 1))
                for m3 in range(3):
                    for m2 in range(3):
                        for imo in range(num_mos):
                            sp_block[0, m3, 0] += (CG011[0, m2, m3] * C[iao1, imo] * C_eps[iao2+1+m2, imo])
                sp_pair_features.append(sp_block)
                ps_pair_features.append(sp_block)

            elif elem1 != "H" and elem2 == "H":

                sp_block = np.zeros((1, 3, 1))
                for m3 in range(3):
                    for m1 in range(3):
                        for imo in range(num_mos):
                            sp_block[0, m3, 0] += (CG011[m1, 0, m3] * C[iao1+1+m1, imo] * C_eps[iao2, imo])
                sp_pair_features.append(sp_block)
                ps_pair_features.append(sp_block)
            
            iao2 += NAO_minimal_valence[elem2]

        iao1 += NAO_minimal_valence[elem1]

    ss_pair_features = np.stack(ss_pair_features)
    sp_pair_features = np.stack(sp_pair_features)
    ps_pair_features = np.stack(ps_pair_features)
    pp_pair_features = np.stack(pp_pair_features)     

    return ss_pair_features, sp_pair_features, ps_pair_features, pp_pair_features



def pair_features(H, elems):
    CG = clebsch_gordan(1, 1, 2)
    CG011 = CG[0, 1:, 1:4]
    CG101 = CG[1:, 0, 1:4]
    CG11 = CG[1:, 1:, :]

    eps, C = np.linalg.eig(H)
    sort_idx = np.argsort(eps)[::-1]
    C = C[:, sort_idx]
    eps = eps[sort_idx]
    C_eps = np.dot(C, np.diag(eps))

    ss_pair_features = []
    sp_pair_features = []
    ps_pair_features = []
    pp_pair_features = []

    iao1 = 0

    for ielem1, elem1 in enumerate(elems):
    
        iao2 = 0

        for elem2 in elems[:ielem1]:

            ss_block = np.zeros((1, 1, 1))
            ss_block[0, 0, 0] = np.dot(C[iao1, :], C_eps[iao2, :])
            ss_pair_features.append(ss_block)

            if elem1 != "H" and elem2 != "H":
                    
                sp_block = np.zeros((1, 3, 1))
                sp_block[0, :, 0] = np.einsum("bc, n, bn -> c", CG011, C[iao1, :], C_eps[iao2+1 : iao2+4, :])
                sp_pair_features.append(sp_block)

                ps_block = np.zeros((1, 3, 1))
                ps_block[0, :, 0] = np.einsum("ac, an, n -> c", CG101, C[iao1+1 : iao1+4, :], C_eps[iao2, :])
                ps_pair_features.append(ps_block)

                pp_block = np.zeros((1, 9, 1))
                pp_block[0, :, 0] = np.einsum("abc, an, bn -> c", CG11, C[iao1+1 : iao1+4, :], C_eps[iao2+1 : iao2+4, :])
                pp_pair_features.append(pp_block)

            elif elem1 == "H" and elem2 != "H":

                sp_block = np.zeros((1, 3, 1))
                sp_block[0, :, 0] = np.einsum("bc, n, bn -> c", CG011, C[iao1, :], C_eps[iao2+1 : iao2+4, :])
                sp_pair_features.append(sp_block)
                ps_pair_features.append(sp_block)

            elif elem1 != "H" and elem2 == "H":
                        
                sp_block = np.zeros((1, 3, 1))
                sp_block[0, :, 0] = np.einsum("ac, an, n -> c", CG101, C[iao1+1 : iao1+4, :], C_eps[iao2, :])
                sp_pair_features.append(sp_block)
                ps_pair_features.append(sp_block)            

            iao2 += NAO_minimal_valence[elem2]

        iao1 += NAO_minimal_valence[elem1]
    
    ss_pair_features = np.stack(ss_pair_features)
    sp_pair_features = np.stack(sp_pair_features)
    ps_pair_features = np.stack(ps_pair_features)
    pp_pair_features = np.stack(pp_pair_features)     

    return ss_pair_features, sp_pair_features, ps_pair_features, pp_pair_features



def atom_features(H, elems):
    CG = clebsch_gordan(1, 1, 2)
    CG011 = CG[0, 1:, 1:4]
    CG11 = CG[1:, 1:, :]

    eps, C = np.linalg.eig(H)
    sort_idx = np.argsort(eps)[::-1]
    C = C[:, sort_idx]
    eps = eps[sort_idx]
    C_eps = np.dot(C, np.diag(eps))

    ss_atom_features = []
    sp_atom_features = []
    pp_atom_features = []

    iao1 = 0

    for elem in elems:

        ss_block = np.zeros((1, 1, 1))
        ss_block[0, 0, 0] = np.dot(C[iao1, :], C_eps[iao1, :])
        ss_atom_features.append(ss_block)

        if elem != "H":

            sp_block = np.zeros((1, 3, 1))
            sp_block[0, :, 0] = np.einsum("bc, n, bn -> c", CG011, C[iao1, :], C_eps[iao1+1 : iao1+4, :])
            sp_atom_features.append(sp_block)

            pp_block = np.zeros((1, 9, 1))
            pp_block[0, :, 0] = np.einsum("abc, an, bn -> c", CG11, C[iao1+1 : iao1+4, :], C_eps[iao1+1 : iao1+4, :])
            pp_atom_features.append(pp_block)

        iao1 += NAO_minimal_valence[elem]
    
    ss_atom_features = np.stack(ss_atom_features)
    sp_atom_features = np.stack(sp_atom_features)
    pp_atom_features = np.stack(pp_atom_features)

    return ss_atom_features, sp_atom_features, pp_atom_features



def get_ao_indices(elems: list) -> dict:
	ao_indices = {}
	aoi = 0
	for i, elem in enumerate(elems):
		if elem == "H":
			ao_indices[f"H{i}"] = {
				"s": (aoi, aoi+1)
			}
			aoi += 1
		else:
			ao_indices[f"{elem}{i}"] = {
				"s": (aoi, aoi+1),
				"p": (aoi+1, aoi+4)
			}
			aoi += 4
	return ao_indices



def get_atom_and_pair_features(H, elems):
        eps, C = np.linalg.eig(H)
        sort_idx = np.argsort(eps)[::-1]
        C = C[:, sort_idx]
        eps = eps[sort_idx]
        C_eps = np.dot(C, np.diag(eps))

        num_atoms = len(elems)
        
        ss_atom_features = np.zeros((1, 1, 1, 1))
        sp_atom_features = np.zeros((1, 1, 1, 1))
        pp_atom_features = np.zeros((1, 1, 9, 1))
        #ss_atom_features = []
        #sp_atom_features = []
        #pp_atom_features = []
        #pair_features = np.zeros((int(num_atoms*(num_atoms-1)/2), 1, 9, 1))
        s_pair_features = np.zeros((num_atoms**2-num_atoms, 1, 1, 1))
        p_pair_features = np.zeros((num_atoms**2-num_atoms, 1, 9, 1))

        ref_p_atom_features = np.zeros((num_atoms, 1, 9, 1))

        CG = clebsch_gordan(1, 1, 2)
        CG011 = CG[0, 1:4, 1:4]
        CG101 = CG[1:4, 0, 1:4]
        #CG = CG[1:, 1:, :]

        iao1 = 0
        ipair = 0

        for ielem1, elem1 in enumerate(elems):
            
            ss_atom_features[ielem1, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao1, :])
            if elem1 != "H":
                pp_atom_features[ielem1, 0, :, 0] = np.einsum("abc, an, bn -> c", CG[1:, 1:, :], C[iao1+1 : iao1+4, :], C_eps[iao1+1 : iao1+4, :])

            iao2 = 0

            for ielem2, elem2 in enumerate(elems):
                
                # THIS COULD BE REPLACED BY FILLING THE ATOM FEATURE TENSOR (MOVING IT HERE FROM THE OUTER LOOP) INSTEAD OF THE PAIR FEATURE TENSOR
                if ielem1 == ielem2:
                     continue
                
                '''
                if elem1 == "H" and elem2 == "H":
                    s_pair_features[ipair, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao2, :])
                elif elem1 != "H" and elem2 != "H":
                    #pair_features[ipair, 0, :, 0] = np.einsum("abc, an, bn -> c", CG, C[iao1 : iao1+4, :], C_eps[iao2 : iao2+4, :])
                    s_pair_features[ipair, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao2, :])
                    p_pair_features[ipair, 0, :, 0] = np.einsum("abc, an, bn -> c", CG, C[iao1+1 : iao1+4, :], C_eps[iao2+1 : iao2+4, :])
                elif elem1 == "H" and elem2 != "H":
                    #pair_features[ipair, 0, :4, 0] = np.dot(C[iao1, :], C_eps[iao2 : iao2+4, :].T)
                elif elem1 != "H" and elem2 == "H":
                    #pair_features[ipair, 0, :4, 0] = np.dot(C[iao1 : iao1+4, :], C_eps[iao2, :])
                    s_pair_features[ipair, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao2, :])
                '''

                s_pair_features[ipair, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao2, :])
                if elem1 != "H" and elem2 != "H":
                    p_pair_features[ipair, 0, :, 0] = np.einsum("abc, an, bn -> c", CG[1:, 1:, :], C[iao1+1 : iao1+4, :], C_eps[iao2+1 : iao2+4, :])
                elif elem1 == "H" and elem2 != "H":
                    #p_pair_features[ipair, 0, 1:4, 0] = np.dot(C[iao1, :], C_eps[iao2+1 : iao2+4, :].T)
                    p_pair_features[ipair, 0, 1:4, 0] = np.einsum("bc, n, bn -> c", CG011, C[iao1, :], C_eps[iao2+1 : iao2+4, :])
                elif elem1 != "H" and elem2 == "H":
                    #p_pair_features[ipair, 0, 1:4, 0] = np.dot(C[iao1+1 : iao1+4, :], C_eps[iao2, :])
                    p_pair_features[ipair, 0, 1:4, 0] = np.einsum("ac, an, n -> c", CG101, C[iao1+1 : iao1+4, :], C_eps[iao2, :])

                iao2 += NAO_minimal_valence[elem2]
                ipair += 1
            
            iao1 += NAO_minimal_valence[elem1]
        
        iao1 = 0
        ipair = 0
        for ielem1, elem1 in enumerate(elems):
            iao2 = 0
            for ielem2, elem2 in enumerate(elems):
                if ielem1 == ielem2:
                    s_atom_features[ielem1, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao1, :])
                    if elem1 != "H":
                        p_atom_features[ielem1, 0, :, 0] = np.einsum("abc, an, bn -> c", CG[1:, 1:, :], C[iao1+1 : iao1+4, :], C_eps[iao1+1 : iao1+4, :])
                else:
                    s_pair_features[ipair, 0, 0, 0] = np.dot(C[iao1, :], C_eps[iao2, :])
                    if elem1 != "H" and elem2 != "H":
                        p_pair_features[ipair, 0, :, 0] = np.einsum("abc, an, bn -> c", CG[1:, 1:, :], C[iao1+1 : iao1+4, :], C_eps[iao2+1 : iao2+4, :])
                    elif elem1 == "H" and elem2 != "H":
                        p_pair_features[ipair, 0, 1:4, 0] = np.einsum("bc, n, bn -> c", CG011, C[iao1, :], C_eps[iao2+1 : iao2+4, :])
                    elif elem1 != "H" and elem2 == "H":
                        p_pair_features[ipair, 0, 1:4, 0] = np.einsum("ac, an, n -> c", CG101, C[iao1+1 : iao1+4, :], C_eps[iao2, :])
                iao2 += NAO_minimal_valence[elem2]
            iao1 += NAO_minimal_valence[elem1]
        
        return s_atom_features, p_atom_features, s_pair_features, p_pair_features



def get_fock_matrix(s_atom_features, p_atom_features, s_pair_features, p_pair_features, elems, num_aos, H_ref):

    s_atom_features = np.squeeze(s_atom_features)
    p_atom_features = np.squeeze(p_atom_features)
    s_pair_features = np.squeeze(s_pair_features)
    p_pair_features = np.squeeze(p_pair_features)

    H = np.zeros((num_aos, num_aos))

    CG = clebsch_gordan(1, 1, 2)
    CG011 = CG[0, 1:4, 1:4]
    CG101 = CG[1:4, 0, 1:4]
    CG = CG[1:, 1:, :]

    iao1 = 0
    ipair = 0

    for ielem1, elem1 in enumerate(elems):

        H[iao1, iao1] = s_atom_features[ielem1]
        if not np.allclose(H_ref[iao1, iao1], H[iao1, iao1]):
            print("hier1")

        if elem1 != "H":
            H[iao1+1:iao1+4, iao1+1:iao1+4] = np.einsum("abc, c -> ab", CG, p_atom_features[ielem1, :])
            if not np.allclose(H_ref[iao1+1:iao1+4, iao1+1:iao1+4], H[iao1+1:iao1+4, iao1+1:iao1+4]):
                print("hier2")

        iao2 = 0

        for ielem2, elem2 in enumerate(elems):
            
            # SELBER HINWEIS WIE OBEN ZU DIESEM CODE BLOCK
            if ielem1 == ielem2:
                continue

            H[iao1, iao2] = s_pair_features[ipair]
            if not np.allclose(H_ref[iao1, iao2], H[iao1, iao2]):
                print("hier3")
            if elem1 != "H" and elem2 != "H":
                H[iao1+1:iao1+4, iao2+1:iao2+4] = np.einsum("abc, c -> ab", CG, p_pair_features[ipair, :])
                if not np.allclose(H_ref[iao1+1:iao1+4, iao2+1:iao2+4], H[iao1+1:iao1+4, iao2+1:iao2+4]):
                    print("hier4")
            elif elem1 == "H" and elem2 != "H":
                #H[iao1, iao2+1:iao2+4] = np.einsum("ab, b -> a", CG011, p_pair_features[ipair, 1:4])
                H[iao1, iao2+1:iao2+4] = np.dot(CG011, p_pair_features[ipair, 1:4])
                if not np.allclose(H_ref[iao1, iao2+1:iao2+4], H[iao1, iao2+1:iao2+4]):
                    print("hier5")
            elif elem1 != "H" and elem2 == "H":
                H[iao1+1:iao1+4, iao2] = np.dot(CG101, p_pair_features[ipair, 1:4])
                if not np.allclose(H_ref[iao1+1:iao1+4, iao2], H[iao1+1:iao1+4, iao2]):
                    print("hier6")

            iao2 += NAO_minimal_valence[elem2]
            ipair += 1

        iao1 += NAO_minimal_valence[elem1]
    
    #'''
    iao1 = 0
    for ielem1, elem1 in enumerate(elems):
        if not np.allclose(H_ref[iao1, iao1], H[iao1, iao1]):
            print("hier1")
        if elem1 != "H":
            if not np.allclose(H_ref[iao1+1:iao1+4, iao1+1:iao1+4], H[iao1+1:iao1+4, iao1+1:iao1+4]):
                print("hier2")
        iao2 = 0
        for ielem2, elem2 in enumerate(elems):
            if ielem1 == ielem2:
                 continue
            if not np.allclose(H_ref[iao1, iao2], H[iao1, iao2]):
                print("hier3")
            if elem1 != "H" and elem2 != "H":
                if not np.allclose(H_ref[iao1+1:iao1+4, iao2+1:iao2+4], H[iao1+1:iao1+4, iao2+1:iao2+4]):
                    print("hier4")
            elif elem1 == "H" and elem2 != "H":
                if not np.allclose(H_ref[iao1, iao2+1:iao2+4], H[iao1, iao2+1:iao2+4]):
                    print("hier5")
            elif elem1 != "H" and elem2 == "H":
                if not np.allclose(H_ref[iao1+1:iao1+4, iao2], H[iao1+1:iao1+4, iao2]):
                    print("hier6")
                pass
            iao2 += NAO_minimal_valence[elem2]
        iao1 += NAO_minimal_valence[elem1]
    #'''
            

    #print(np.allclose(H_ref, H))

    return H
