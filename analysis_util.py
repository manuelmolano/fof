### hierarchical clustering with Pearson correlation
import numpy as np
import generate_pseudo_trials as gpt
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

import scipy
import scipy.cluster.hierarchy as sch

def compute_covariance_matrix(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files, type, DOREVERSE=0, CONTROL = 0, n_iterations=10, N_pseudo_dec=25): 
    CovXdata_c = {} 
    CovXdata_e = {} 
    for i in range(n_iterations): 
        Xmerge_hist_trials_correct,ymerge_hist_labels_correct,Xmerge_hist_trials_error,ymerge_hist_labels_error,merge_trials_hist=gpt.merge_pseudo_hist_trials(Xdata_hist_set,ylabels_hist_set,unique_states,unique_cohs,files,false_files,N_pseudo_dec,REC_TRIALS_SET,[])
        Xdata_c   = Xmerge_hist_trials_correct[4][:,:] 
        ylabels_c = ymerge_hist_labels_correct[4][:,:]
        Xdata_e   = Xmerge_hist_trials_error[0][:,:] 
        ylabels_e = ymerge_hist_labels_error[0][:,:] 
        for state in range(1,4):
   	        Xdata_c   = np.vstack((Xdata_c,Xmerge_hist_trials_correct[state+4][: ,:]))
   	        ylabels_c = np.vstack((ylabels_c,ymerge_hist_labels_correct[state+4][:,:]))
   	     
   	        Xdata_e   = np.vstack((Xdata_e,Xmerge_hist_trials_error[state][:,:]))
   	        ylabels_e = np.vstack((ylabels_e,ymerge_hist_labels_error[state][:,:]))
        Xdata_c_subm = Xdata_c-np.repeat(np.mean(Xdata_c,axis=0,keepdims=True),np.shape(Xdata_c)[0],axis=0) 
        Xdata_c_subm = Xdata_c_subm[np.where(np.abs(Xdata_c_subm)>0)[0]]
        Xdata_e_subm = Xdata_e-np.repeat(np.mean(Xdata_e,axis=0,keepdims=True),np.shape(Xdata_c)[0],axis=0)
        const_c = np.where(np.std(Xdata_c_subm,axis=0)>0)[0]
        const_e = np.where(np.std(Xdata_e_subm,axis=0)>0)[0]
        const   = np.intersect1d(const_c,const_e)
        Xdata_c_subm = Xdata_c_subm[:,const]
        CovXdata_c[i]=np.corrcoef(Xdata_c_subm.T)
        Xdata_e_subm = Xdata_e_subm[:,const] 
        CovXdata_e[i]=np.corrcoef(Xdata_e_subm.T)
    return CovXdata_c, CovXdata_e

NITERATIONS = 50
REC_TRIALS_SET = {}
for itr in range(NITERATIONS):
    REC_TRIALS_SET[itr] = {}
Xdata_hist_set, ylabels_hist_set, files = data_tr['Xdata_hist_set'], data_tr['ylabels_hist_set'], data_tr['files']

Xcohs_0 = data_tr['Xcohs_0']
unique_states = np.arange(8)
unique_cohs = np.sort(Xcohs_0)
CovXdata_c,CovXdata_e=compute_covariance_matrix(Xdata_hist_set, ylabels_hist_set,unique_states, unique_cohs, files,false_files, type, DOREVERSE=DOREVERSE,CONTROL=CONTROL, n_iterations=NITERATIONS,N_pseudo_dec=25)

idx_iter = 16
n_variable = np.shape(CovXdata_e[idx_iter])[0]
C=CovXdata_e[idx_iter][:,:]-np.eye(n_variable)


X = C
d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ind = sch.fcluster(L, 0.5*d.max(), 'distance')
columns = [i for i in list((np.argsort(ind)))]

XX,YY = np.meshgrid(columns,columns)
plt.figure()
# C=CovXdata_c[idx_iter][:,:]-np.eye(n_variable)
# C=CovXdata_c[idx_iter][:,:]-np.eye(n_variable)
plt.imshow(C[XX,YY], cmap='seismic',vmin=-0.3,vmax=0.3)

plt.figure()
# C=CovXdata_c[idx_iter][:,:]-np.eye(n_variable)
C=CovXdata_c[idx_iter][:,:]-np.eye(n_variable)
plt.imshow(C[XX,YY], cmap='seismic',vmin=-0.3,vmax=0.3)




# # This generates 100 variables that could possibly be assigned to 5 clusters
# idx_iter=10
# n_variables = np.shape(CovXdata_c[idx_iter])[0]
# n_clusters  = 4

# # To keep this example simple, each cluster will have a fixed size
# cluster_size = n_variables // n_clusters

# # Assign each variable to a cluster
# belongs_to_cluster = np.repeat(range(n_clusters), cluster_size)
# np.random.shuffle(belongs_to_cluster)


# variables = np.array(variables)

# C=CovXdata_c[idx_iter][:,:]-np.eye(n_variables)
# # C=CovXdata_e[idx_iter][:,:]-np.eye(n_variables)
# initial_C = C
# initial_score = score(C)
# initial_ordering = np.arange(n_variables)

# plt.figure()
# plt.imshow(C, interpolation='nearest')
# plt.title('Initial C')
# print('Initial ordering:', initial_ordering)
# print('Initial covariance matrix score:', initial_score)


# current_C = C
# current_ordering = initial_ordering
# current_score = initial_score

# max_iter = 50
# for i in range(max_iter):
#     # Find the best row swap to make
#     best_C = current_C
#     best_ordering = current_ordering
#     best_score = current_score
#     for row1 in range(n_variables):
#         for row2 in range(n_variables):
#             if row1 == row2:
#                 continue
#             option_ordering = best_ordering.copy()
#             option_ordering[row1] = best_ordering[row2]
#             option_ordering[row2] = best_ordering[row1]
#             option_C = swap_rows(best_C, row1, row2)
#             option_score = score(option_C)

#             if option_score > best_score:
#                 best_C = option_C
#                 best_ordering = option_ordering
#                 best_score = option_score

#     if best_score > current_score:
#         # Perform the best row swap
#         current_C = best_C
#         current_ordering = best_ordering
#         current_score = best_score
#     else:
#         # No row swap found that improves the solution, we're done
#         break

# # Output the result
# plt.figure()
# plt.imshow(current_C, interpolation='nearest',cmap='seismic',vmin=-0.4,vmax=0.4)
# plt.title('Best C')
# print('Best ordering:', current_ordering)
# print('Best score:', current_score)
# print
# print('Cluster     [variables assigned to this cluster]')
# print('------------------------------------------------')
# for cluster in range(n_clusters):
#     print('Cluster %02d  %s' % (cluster + 1, current_ordering[cluster*cluster_size:(cluster+1)*cluster_size]))


# # Output the result
# XX,YY = np.meshgrid(best_ordering,best_ordering)
# plt.figure()
# plt.imshow(CovXdata_e[idx_iter][XX,YY], interpolation='nearest',cmap='seismic',vmin=-0.4,vmax=0.4)
# plt.title('Best C')
# print('Best ordering:', current_ordering)
# print('Best score:', current_score)
# print
# print('Cluster     [variables assigned to this cluster]')
# print('------------------------------------------------')
# for cluster in range(n_clusters):
#     print('Cluster %02d  %s' % (cluster + 1, current_ordering[cluster*cluster_size:(cluster+1)*cluster_size]))


IDX_RAT='Rat15_'
### compute the angles between history encoding axises
dataname  = dir+IDX_RAT+'data_dec_ae.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_ae, bi_ae = data_dec['coefs_correct'], data_dec['intercepts_correct']

dataname  = dir+IDX_RAT+'data_dec_ac.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_ac, bi_ac = data_dec['coefs_correct'], data_dec['intercepts_correct']

dataname  = dir+IDX_RAT+'data_dec.npz'
data_dec  = np.load(dataname, allow_pickle=True)
wi_comp, bi_comp = data_dec['coefs_correct'], data_dec['intercepts_correct']

wiac,wiae,wicomp = np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS)),np.zeros((np.shape(wi_ae)[0],NITERATIONS))
for i in range(NITERATIONS):
	wiac[:,i],wiae[:,i],wicomp[:,i] = wi_ac[:, i*5+3],wi_ae[:, i*5+3],wi_comp[:, i*5+3]

wiac, wiae, wicomp= np.mean(wiac,axis=1),np.mean(wiae,axis=1),np.mean(wicomp,axis=1)
ag_ce, ag_cpc, ag_cpe = 0,0,0  
unit_ac=wiac/np.linalg.norm(wiac)
unit_ae=wiae/np.linalg.norm(wiae)
unit_comp=wicomp/np.linalg.norm(wicomp)

ce_dotprod=np.dot(unit_ac,unit_ae)
ag_ce += np.arccos(ce_dotprod)

cpc_dotprod=np.dot(unit_ac,unit_comp)
ag_cpc += np.arccos(cpc_dotprod)

cpe_dotprod=np.dot(unit_ae,unit_comp)
ag_cpe += np.arccos(cpe_dotprod)

print('ae: ',ag_ce*180/np.pi)
print('cpc, cpe: ',ag_cpc*180/np.pi," ",ag_cpe*180/np.pi)

# # Rat32
# ae:  87.95303533426575
# cpc, cpe:  59.0683463688086   39.95837745974393

# # Rat15
# ae:  89.46429263445268
# cpc, cpe:  50.707924537394774   47.833616757836104


wi_stim = data_dec['coefs_correct']
wi_beh  = data_dec['coefs_correct']
wistim = np.zeros((np.shape(wi_stim)[0],NITERATIONS))
wibeh  = np.zeros((np.shape(wi_beh)[0],NITERATIONS))
for i in range(NITERATIONS):
	wistim[:,i]= wi_stim[:, i*5+3]
	wibeh[:,i] = wi_beh[:,i*5+4]

ag_cs, ag_es, ag_cps = 0,0,0
wistim = np.mean(wi_stim,axis=1)
unit_stim=wistim/np.linalg.norm(wistim)

cs_dotprod=np.dot(unit_ac,unit_stim)
ag_cs += np.arccos(cs_dotprod)

es_dotprod=np.dot(unit_ae,unit_stim)
ag_es += np.arccos(es_dotprod)

cps_dotprod=np.dot(unit_comp,unit_stim)
ag_cps += np.arccos(cps_dotprod)

print('stimulus:  cps, cs, es: ',ag_cps*180/np.pi," ",ag_cs*180/np.pi," ",ag_es*180/np.pi)

ag_cs, ag_es, ag_cps = 0,0,0
wibeh = np.mean(wi_beh,axis=1)
unit_beh=wibeh/np.linalg.norm(wibeh)

cs_dotprod=np.dot(unit_ac,unit_beh)
ag_cs += np.arccos(cs_dotprod)

es_dotprod=np.dot(unit_ae,unit_beh)
ag_es += np.arccos(es_dotprod)

cps_dotprod=np.dot(unit_comp,unit_beh)
ag_cps += np.arccos(cps_dotprod)

print('beh:  cps, cs, es: ',ag_cps*180/np.pi," ",ag_cs*180/np.pi," ",ag_es*180/np.pi)

stim_beh_dotprod=np.dot(unit_stim,unit_beh)
ag_stim_beh = np.arccos(stim_beh_dotprod)
print('beh-stim ',ag_stim_beh*180/np.pi)



# cps, cs, es:  88.06160141663162   84.1526362323566   90.85343789128409 # ac Rat15
# cps, cs, es:  84.15824646269478   84.48816975528712   85.72723980926808  #comp 
# # Rat 32
# cps, cs, es:  80.3094365194275   70.11230483439101   88.52403741093009. # ac Rat32
# cps, cs, es:  78.93176061401846   70.62851720897406   85.950307743718.  # comp






