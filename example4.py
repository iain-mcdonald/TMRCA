import numpy as np   # Needed for arrays
import scipy.special # Needed for binomial combinations
import scipy.stats   # Needed for Poisson distributions
import itertools     # Needed for Q iteration in STRs
import random        # Needed for occasionally randomising inputs
import csv           # Needed for output files
from math import log10
from math import sqrt

# ====================================
# Functions defining TMRCA calculation
# ====================================

# Generate weighting factors for a specific genetic distance (g), number of mutations (m) and multi-step combination (Q)
# Calculates the three-line equation for \varpi_g,m[Q] in the accompanying LaTeX document
# Each factors[] represents one of the factors in the equation
# countq = number of multi-step mutations in Q+ (latterly Q-) which have multi-step degree i
def gdq_varpi(g,m,wp,wn,omegap,omegan,qp,qn,kp,kn,epsp,epsn,nqp,nqn,maxnq):
	factors=np.ones(8,dtype=float)
	factors[0]=scipy.special.comb(m,kp,exact=True) # First line - probability of getting kp positives and kn negatives
	factors[1]=wp**kp*wn**kn
	for i in range(maxq): # This performs the loop over multi-step size n
		countq=len(qp[np.where(qp==i)]) # Second line - product of all the probabilities of getting Q multi-step mutations from kp positives
		if (countq>0):
			factors[2]*=scipy.special.comb(kp,countq,exact=True)
			factors[3]*=(omegap[i]/wp)**countq
			factors[4]*=(1-omegap[i]/wp)**(kp-countq)
			#print ("i,countq,kp:",i,countq,kp,scipy.special.comb(kp,countq))
		countq=len(qn[np.where(qn==i)]) # Third line - and probability of getting Q multi-step mutations from kn negatives
		if (countq>0):
			factors[5]*=scipy.special.comb(kn,countq,exact=True)
			factors[6]*=(omegan[i]/wn)**countq
			factors[7]*=(1-omegan[i]/wn)**(kn-countq)
			#print ("i,countq,kn:",i,countq,kn,scipy.special.comb(kn,countq))
	varpi=np.prod(factors) # Perform final multiplication
	#if (varpi>1.e-3):
	#	trace(10, 'g={} m={} q={},{} eps={},{} k={},{} nq={},{} varpi={}'.format(g,m,qp,qn,epsp,epsn,kp,kn,nqp,nqn,varpi))
	return varpi

# Generate weighting factors for obtaining a genetic distance g, given m mutations
# (\varpi_g,m in accompanying LaTeX document)
def gd_weight(g,m,wp,wn,omegap,omegan,maxq,maxnq):
	if (g==0 and m==0): # Special case if no mutations
		varpi=1.
	else: # If some mutations
		varpi=0.
		# General a set of possible multi-step mutations and loop over those sets
		qp0=np.ones(m,dtype=int) # Set up lists of positive/negative mutations
		qn0=np.ones(m,dtype=int)
		qsets=list(itertools.combinations_with_replacement(range(1,maxq),min(maxnq,m)))
		for qpset in qsets:
			for qnset in qsets:
				qp=qp0 # Get original sets of Q+ and Q-
				qn=qn0
				multip=np.array(list(reversed(qpset))) # Generate lists of Q+ and Q-
				multin=np.array(list(reversed(qnset)))
				qp[:len(multip)]=multip
				qn[:len(multip)]=multin
				nqp=len(qp[np.where(qp>1)]) # Count number of multistep mutations (positive & negative)
				nqn=len(qn[np.where(qn>1)])
				if (nqp+nqn<=maxnq): # Only use those where combined Q+ and Q- < maximum no. allowed multi-step mutations
					epsp=np.sum(qp[np.where(qp>1)])-nqp  # Calculate additional repeats caused by multi-step mutations (\varepsilon)
					epsn=np.sum(qn[np.where(qn>1)])-nqn
					kp=(m+g-epsp+epsn)/2. # Calculate number of required positive and negative mutations (k_+, k_-)
					kn=(m-g+epsp-epsn)/2.
					# If an positive and integer number of +/- mutations, if the required genetic distance is obtained, if the number of multistep markers is less than the total
					if (kp>=0 and kn>=0 and kp==int(kp) and kn==int(kn) and kp+kn==m and kp+epsp-kn-epsn==g and nqp<=kp and nqn<=kn):
						qp=qp[:int(kp)]
						qn=qn[:int(kn)]
						varpi+=gdq_varpi(g,m,wp,wn,omegap,omegan,qp,qn,int(kp),int(kn),int(epsp),int(epsn),nqp,nqn,maxnq) # Calculate probability weight
	return varpi

# This returns the probability density functions for a generic Y-STR of genetic distance (g) in mutation timescales
# This calculates p_g(\bar{m}_s) in the accompanying LaTeX document
# ***Note: still need to code in the case where g is negative
# ***where wp==wn, gmweightp = gmweightn
# ***where wp!=wn a new computation is needed
def generate_genpdfstr(mtimes,pdf_str_m,maxg,maxm,maxq,maxnq,maxnm,mres,omegap,omegan):
	gmweight=np.zeros((maxg,maxm)) # Weight generation
	wp=np.sum(omegap[1:]) # Calculate w+ and w-, the weight of obtaining a positive/negative result overall
	wn=np.sum(omegan[1:])
	# Loop over genetic distances and numbers of mutations
	for g in range(maxg):
		for m in range(maxm):
			gmweight[g,m]=gd_weight(g,m,wp,wn,omegap,omegan,maxq,maxnq) # Get weights
			if (gmweight[g,m]>0): # If weight is non-zero
				#trace (3, 'weight[{},{}]={}'.format(g,m,gmweight[g,m]))
				for i in range(len(mtimes)): # Add Poisson distribution of mean m * appropriate weight to overall PDF
					#print(int(m),mtimes[i],scipy.stats.poisson.pmf(int(m),mtimes[i]))
					pdf_str_m[g,i]+=gmweight[g,m]*scipy.stats.poisson.pmf(int(m),mtimes[i])
	#Outputs first few results for inspection
	#np.savetxt("str_pdf.dat", np.c_[mtimes,pdf_str_m[0,:],pdf_str_m[1,:],pdf_str_m[2,:],pdf_str_m[3,:]], delimiter=' ')

# This calculates the probability mass functions for speicifc Y-STRs of genetic distance (g) in physical timescales (generations)
# This calculates p_g(t) in the accompanying LaTeX document
def generate_pdfstr(mtimes,times,pdf_str_m,pdf_str_t,mustr,mustrunc,maxtime,maxg):
	for i in range(len(mustr)):
		# Generate distribution in mu
		#muscale=np.arange(0,maxtime,0.5*mustr[i])
		#mu=np.zeros(len(muscale))
		
		# Need to compute time [t] = mutations [m] * mutation rate [mu]
		# Both m and mu are PDFs, to give t as a PDF
		# Seemingly no functional way of computing the product of two PDFs in Python
		# So let's do this ourselves
		# Not doing this as a function because the implementation for SNPs is subtly different
		# to reduce computational timeframe here
		# Let's represent mu by a log-normal distribution
		logmu=log10(1/mustr[i])
		sigma=log10(1/(1-mustrunc[i]/mustr[i]))
		#trace (5, 'STR {} of {}: mu={} yr unc={}% logmean={} logsigma={}'.format(i+1,len(mustr),1/mustr[i],mustrunc[i]/mustr[i]*100,logmu,sigma))
		# Sample points from this log-normal distribution
		ppfsamp=np.arange(0.02,0.98,0.04) # This samples 24 points - more may be required if fractional uncertainty is large
		ppfsamps=len(ppfsamp)
		lnsamp=scipy.stats.lognorm.ppf(ppfsamp,sigma,logmu-1)
		# Then translate m to t using that sampled mu
		for musamp in lnsamp:
			xtimes=mtimes*10**musamp
			for g in range(maxg):
				pdf_str_t[i,g,:]+=np.interp(times,xtimes,pdf_str_m[g,:])/ppfsamps
				#print (g,10**musamp,xtimes,pdf_str_t[i,g,:])
		#Outputs first few results for inspection
		#np.savetxt("str_pdf_t.dat", np.c_[times,pdf_str_t[0,0,:],pdf_str_t[0,1,:],pdf_str_t[0,2,:],pdf_str_t[0,3,:]], delimiter=' ')
				

# STR TMRCA calculator for one test
def strage(times,pdf_str_t,ancht,derht):
	nstr=min(len(ancht),len(derht)) # Total number of STRs
	gd=abs(ancht-derht) # Find GD on each STR *** Remove abs to allow negative genetic distances if wp!=wn
	#trace (5, 'GD:{}'.format(gd))
	tmrca_str=np.ones(len(times)) # Setup TMRCA array
	strcomp=0
	for i in range(nstr): # Loop over STRs
		if (ancht[i]>0 and derht[i]>0): # If non-null
			strcomp+=1
			tmrca_str*=pdf_str_t[i,gd[i],:] # *** Needs edit to allow negative GD for wp!=wn
	# Normalise
	psum=sum(tmrca_str)
	tmrca_str/=psum
	#trace(5, '{} STRs compared'.format(strcomp))
	#Output result for inspection
	#np.savetxt("tmrca_str.dat", np.c_[times,tmrca_str], delimiter=' ')
	return tmrca_str
			

# SNP TMRCA calculator for one test
def snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc):
	pdf_snp=np.zeros(len(times))
	# As elsewhere, we need to compute time [t] = mutations [m] * mutation rate [mu]
	# Both m and mu are PDFs, to give t as a PDF
	# Let's represent mu by a log-normal distribution
	logmu=log10(1/testcombcov/musnp)
	sigma=log10(1/(1-musnpunc/musnp))
	#trace (5, 'SNPs: cov={} mu={} cov*mu={} yr/SNP unc={}% logmean={} logsigma={}'.format(testcombcov,musnp,1/(testcombcov*musnp),musnpunc/musnp*100,logmu,sigma))
	# Sample points from this log-normal distribution
	ppfsamp=np.arange(0.02,0.98,0.04) # This samples 24 points - more may be required if fractional uncertainty is large
	ppfsamps=len(ppfsamp)
	lnsamp=scipy.stats.lognorm.ppf(ppfsamp,sigma,logmu-1)
	# Then translate m to t using that sampled mu
	for musamp in lnsamp:
		xtimes=stimes*10**musamp
		pdf_snp+=np.interp(times,xtimes,tsnpm)/ppfsamps
		#print (10**musamp,xtimes,pdf_snp)
	#Outputs first few results for inspection
	#np.savetxt("snp_pdf_t.dat", np.c_[times,pdf_snp,], delimiter=' ')
	return pdf_snp
	
# Paper trail estimator
def paperpdf(dates,pdftype,t,s=1,a=1,psi=0):
	pdf=np.ones(len(dates))
	if (pdftype=="delta"):
		pdf=psi
		idx=(np.abs(dates-t)).argmin()
		pdf[idx]=1
	elif (pdftype=="smooth"):
		pdf=scipy.stats.norm.pdf((t-dates)/s)*(1-psi)+psi
	elif (pdftype=="step-up"):
		pdf[dates>t]=psi
	elif (pdftype=="step-down"):
		pdf[dates<t]=psi
	elif (pdftype=="smooth-start"):
		pdf=scipy.stats.norm.cdf((t-dates)/s)*(1-psi)+psi
	elif (pdftype=="smooth-end"):
		pdf=scipy.stats.norm.cdf((dates-t)/s)*(1-psi)+psi
	elif (pdftype=="ln-start"):
		pdf=scipy.stats.lognorm.cdf((t-dates)/a,log10(s))*(1-psi)+psi
	elif (pdftype=="ln-end"):
		pdf=scipy.stats.lognorm.cdf((dates-t)/a,log10(s))*(1-psi)+psi
	else:
		# May ultimately want to interpret this as a filename
		print ("Paper trail PDF type not recognised:{}".format(pdftype))
	return pdf
	
# Ages to dates calculator
def ages2dates(pdf_times,dates,times,nowtime,t0,st):
	# As elsewhere, we need to compute dates [dates] = zero [t0] - times [t]
	# Both m and mu are PDFs, to give t as a PDF
	# Let's represent zero by a normal distribution
	pdf_zero=scipy.stats.norm.pdf((times-(nowtime-t0))/st)
	# Convolve times with zero to get dates
	pdf_dates=scipy.convolve(pdf_times,pdf_zero,mode='full')
	pdf_dates=pdf_dates[:len(dates)]
	return pdf_dates
	
def test2(pdmrca1,pdmrca2,ht0,ht1,ht2,testsnpdist1,testsnpdist2,testcombcov,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper):
	# Perform the TMRCA calculation for two test results
    # Inputs (prior date to MRCA 1, prior date to MRCA 2, ancestral haplotype, halpotype1, haplotype2, #SNPs (1, 2), combined coverage, paper trail (year, uncertainty, alpha parameter and constraint type) (1, 2), paper trail surname contraint type, flags for doing STRs, SNPs, InDels and paper trails
    # Types  (NumPy arrays...                              integers...                                                                          strings...                                                   boolean integer (0/1)...
		
	if (do_paper>0):
		tmrca_paper=np.ones(len(dates))
		# TMRCA from paper results from the conjunction of TMRCA 1 & 2,
		# so it is the maximum probability of being before this date.
		# This allows for one line to join the other before the oldest
		# most-distant known ancestor.
		tmrca_paper*=np.maximum(paperpdf(dates,testpapertype1,testpaper1,testpaperunc1,testpaperalpha1),paperpdf(dates,testpapertype2,testpaper2,testpaperunc1,testpaperalpha1))
		# Account for shared surname
		tmrca_paper*=paperpdf(dates,testpapertypes,testpapers,testpaperuncs)
	else:
		tmrca_paper=np.ones(len(dates))
	
	if (do_strs>0):
		tstr1=strage(times,pdf_str_t,ht0,ht1)
		tstr2=strage(times,pdf_str_t,ht0,ht2)
	else:
		tstr1=np.ones(len(times))
		tstr2=np.ones(len(times))

	if (do_snps>0):
		stimes=np.arange(0,maxsnps,snpstep)
		tsnpm1=scipy.stats.poisson.pmf(testsnpdist1,stimes)
		tsnpm2=scipy.stats.poisson.pmf(testsnpdist2,stimes)
		# Convert to physical timescale
		tsnp1=snptmconv(times,stimes,tsnpm1,testcombcov,musnp,musnpunc)
		tsnp2=snptmconv(times,stimes,tsnpm2,testcombcov,musnp,musnpunc)
	else:
		tsnp1=np.ones(len(times))
		tsnp2=np.ones(len(times))
	
	if (do_indels>0):
		trace(0, 'Calculating indel-based TMRCA')
		stimes=np.arange(0,maxsnps,snpstep)
		tsnpm1=scipy.stats.poisson.pmf(testindeldist1,stimes)
		tsnpm2=scipy.stats.poisson.pmf(testindeldist2,stimes)
		# Convert to physical timescale
		tindel1=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
		tindel2=snptmconv(times,stimes,tsnpm,testcombcov,musnp,musnpunc)
	else:
		tindel1=np.ones(len(times))
		tindel2=np.ones(len(times))

	#tmrca=tmrca_str*tmrca_snp*tmrca_indel
	tmrca1=tstr1*tsnp1*tindel1
	tmrca2=tstr2*tsnp2*tindel2
	tmrca=tmrca1*tmrca2
	# Normalise PDF
	tmrca/=np.sum(tmrca)
	tmrca1/=np.sum(tmrca1)
	tmrca2/=np.sum(tmrca2)
	# Convert to dates
	dt=dt0/sqrt(2) # Average birth date uncertainty is reduced by factor sqrt(2) for two testers
	st=sqrt(st0**2+dt**2) # Add uncertainty in the mean
	dmrca=ages2dates(tmrca,dates,times,nowtime,t0,st) # Convert TMRCA to dates
	dmrca1=np.convolve(pdmrca1,tmrca1,mode='full')[0:len(tmrca1)]
	dmrca2=np.convolve(pdmrca2,tmrca2,mode='full')[0:len(tmrca2)]
	dmrca=dmrca1*dmrca2
	dmrca*=tmrca_paper # Add in paper-trail genealogy limits
	dmrca1*=tmrca_paper
	dmrca2*=tmrca_paper
	# Normalise PDF
	psum=sum(dmrca)
	dmrca/=psum

	return dmrca,dmrca1,dmrca2,tmrca,tmrca1,tmrca2


# ====================================================
# Main code - set global parameters for use in example
# ====================================================

# Zero points
t0 = 1956.0
st0 = 1.0 # Uncertainty in the mean
dt0 = 10.0 # Standard deviation
# STR rates and uncertainties
mustr=np.array([0.001776,0.003558,0.002262963333333,0.00214183,0.001832,0.003458,0.0002157006,0.000537249333333,0.003600416666667,0.002361616666667,0.000516113666667,0.0026555,0.007048133333333,0.000523,0.001525,0.000868596333333,0.000918865,0.0019482,0.001313576666667,0.001448,0.0089895,0.001874,0.0032,0.004241,0.003716,0.002626356666667,0.0022795,0.000496,0.001141,0.004139643333333,0.002660216666667,0.0096684,0.006975733333333,0.014357,0.018449,0.002620776666667,0.000612560666667,0.001436,0.000990948666667,0.000433,0.000319,0.000254032,0.00166647,0.000919199333333,0.0002167,0.00282426,0.0017663,0.000236,0.00229,0.001527,0.0034075,0.000467199,0.000203582333333,0.0003603,0.005522,0.000264693666667,0.0027805,0.005328563333333,0.0020715,0.0028765,0.00055675,0.001144006333333,0.00081927,0.001894,0.000297,0.000300432666667,0.001237244666667,0.018279,0.00106261,6.96637E-05,0.000826361,0.001597966666667,0.007726,0.001002,0.00064516,0.00218298,0.001634593333333,0.004000833333333,0.0007529,0.002278673333333,0.000260055333333,0.0027318,0.001357,0.000812522666667,0.0013825,0.001339923333333,0.00105415,0.001369627,0.00373403,0.00076,0.002423076666667,0.00084,0.00178172,0.016378,0.000278238,0.007583,0.004142,0.00387799,0.006949,0.0024505,0.00176344,0.00301,0.0002895845,0.003112,0.0008633,0.00133475,0.0007121,0.0026829,0.001059957666667,0.002533206666667,0.000909901666667,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000866329393708,0.000646939655185,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000185879225554,1.00E-06,1.00E-06,1.00E-06,7.18628072330345E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.30389844826262E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.31170729064071E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,5.77076443913719E-05,7.19120620701156E-06,1.00E-06,1.00E-06,1.00E-06,2.88041462217885E-05,1.00E-06,7.17644996995205E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,7.20107745785662E-06,7.19613844723963E-06,1.00E-06,1.00E-06,1.00E-06,0.000230751008629,1.00E-06,1.43627104792525E-05,6.48782185068035E-05,7.22090148387681E-06,1.00E-06,6.50328710086277E-05,8.6689637041137E-05,1.00E-06,0.001323535604723,1.00E-06,1.00E-06,1.00E-06,1.00E-06,2.16477441055493E-05,7.20602325276963E-06,1.00E-06,1.00E-06,1.43823988732946E-05,1.00E-06,1.00E-06,0.008133689075989,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.44120328808033E-05,0.000178671921343,1.00E-06,1.44716687654042E-05,1.00E-06,7.20107745785662E-06,8.62299433967952E-05,4.33247885674273E-05,1.00E-06,1.00E-06,2.90634603345777E-05,1.00E-06,0.003577001096869,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,7.230854483894E-06,7.21593525175012E-06,1.00E-06,1.00E-06,7.34217624636386E-06,1.00E-06,1.00E-06,7.32679459221162E-06,7.26590710251738E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000340653719367,1.00E-06,7.49426418290361E-06,1.00E-06,8.71104600989299E-05,7.23584128005022E-06,7.23085448384135E-06,1.00E-06,7.26087879312772E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000107848004839,1.00E-06,0.00070976761243,0.002858815116904,1.00E-06,2.16031711042177E-05,1.00E-06,7.25084302421095E-06,7.311477251621E-06,0.001408707269388,1.00E-06,1.00E-06,0.001773519683037,1.00E-06,0.000115613331671,1.00E-06,1.00E-06,0.000156930073995,1.00E-06,1.00E-06,1.00E-06,2.16180083626181E-05,1.00E-06,2.16626426743719E-05,7.21097584608153E-06,0.000264790198753,6.55725075769289E-05,1.4501672167711E-05,0.000957091249245,1.00E-06,1.00E-06,7.23584128005022E-06,0.004884093197363,5.80669449264951E-05,7.23085448384135E-06,0.000375542091327,9.32869226725564E-05,7.39913247958075E-06,1.00E-06,2.89234179353917E-05,1.00E-06,1.45822928440098E-05,0.000240161874603,1.00E-06,1.45318142050708E-05,2.2098698583111E-05,0.00054974441262,1.00E-06,1.00E-06,1.00E-06,0.000814739188919,0.001692263028863,7.29115347885296E-06,1.00E-06,0.00029993372441,0.000151842964233,1.00E-06,3.04115068291627E-05,1.00E-06,0.002179366480775,1.00E-06,0.001176200984885,1.00E-06,3.62038277773263E-05,0.000228878374873,1.00E-06,1.00E-06,7.28609017780372E-06,0.000695661057329,0.002038775146703,0.00034583869351,2.17325060312092E-05,0.000202042136976,2.90837695247712E-05,0.000224370527201,1.00E-06,2.91039385743883E-05,1.00E-06,7.26892826973318E-05,1.00E-06,7.2962238220587E-06,1.00E-06,0.000479061056326,1.00E-06,7.21097584608153E-06,1.00E-06,0.000345084936576,1.00E-06,6.48194321241704E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,4.3275773887849E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,2.22078408156419E-05,1.00E-06,1.00E-06,7.311477251621E-06,0.000137443809509,1.00E-06,1.00E-06,7.62497809308834E-06,0.000228722497731,2.17224425871507E-05,1.4501672167711E-05,2.31383827594682E-05,2.17025370439257E-05,1.45924335008614E-05,1.00E-06,2.2514887404072E-05,1.00E-06,0.000727471884019,0.004627006466369,0.000908822480141,1.00E-06,1.00E-06,1.00E-06,6.13784627205385E-05,1.00E-06,1.00E-06,0.000253104032158,1.00E-06,1.00E-06,1.00E-06,0.005655396000866,1.00E-06,0.002290119886728,1.00E-06,1.00E-06,0.000141953388767,1.00E-06,0.002976316037582,0.000794431508604,7.32168168604791E-06,1.00E-06,0.000120857824116,1.00E-06,0.001113326336591,1.00E-06,1.00E-06,0.000152916056261,1.00E-06,0.005242230390096,1.00E-06,2.34661382616262E-05,7.26590710251738E-06,2.23708698150525E-05,1.00E-06,4.73440664708861E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000327325431358,0.000122996751763,1.00E-06,1.00E-06,8.73612992481482E-05,0.00153550001202,2.21921831100369E-05,7.24083495932796E-06,0.000376992758557,1.00E-06,1.00E-06,2.16577382194611E-05,8.10249390867033E-05,0.008714410639185,0.000829703192635,0.000872147285301,7.60839003340416E-06,1.00E-06,0.004355995076024,2.28914976825357E-05,0.001977045000582,0.002592997087187,1.00E-06,1.00E-06,2.21609338427711E-05,1.52057374107413E-05,0.004876561421801,0.001447751657171,0.001501338426772,7.43583972788376E-06,0.000896489821155,1.00E-06,1.00E-06,7.19613844723963E-06,1.00E-06,1.00E-06,1.00E-06,3.62842807279286E-05,1.00E-06,1.00E-06,1.00E-06,7.3887111662081E-06,1.00E-06,0.000125529750045,1.00E-06,1.00E-06,0.000129422649816,0.001287373752605,0.001351076257623,1.00E-06,3.63040440504395E-05,7.32679459221162E-06,0.000123344245185,7.22476883737539E-05,1.00E-06,7.39391815085335E-06,7.87685424626355E-06,0.000159909999226,1.00E-06,7.38351151023684E-06,1.00E-06,3.20853084707844E-05,0.000701370982279,7.75788957522413E-05,7.33704185737334E-06,7.4782393842272E-06,7.40435416800369E-06,0.000994525963611,1.00E-06,7.06549683294134E-05,1.00E-06,0.002030114945391,1.53728496059504E-05,2.99129575368543E-05,0.000473332555761,1.00E-06,0.001022672298961,0.000323892700767,0.00051820765804,0.002086720643268,2.91443607112809E-05,1.00E-06,0.000195652693676,7.69770348940673E-06,1.00E-06,0.000651357389689,0.001469025770727,1.00E-06,5.36365594267711E-05,1.00E-06,0.001009358042687,1.00E-06,0.00093988518356,2.30761053423314E-05,0.002999455874981,1.00E-06,0.001095112066371,1.00E-06,1.00E-06,3.55958566255592E-05,1.00E-06,1.00E-06,1.00E-06,1.00E-06,0.000188971689956,0.001242269874803,1.60182558783611E-05,1.00E-06,4.78348932246706E-05,0.00038676857562,1.00E-06,2.59273318752663E-05,1.53391372164636E-05,7.54275331133574E-06,0.006006121688146,1.00E-06,1.00E-06,1.00E-06,0.002345691789981,1.00E-06,1.00E-06,0.002382318331383,1.00E-06,7.3700230952696E-05,0.000391849483953,1.00E-06,0.001089617317705,0.000111346867817,0.000802564339049,0.000153930897162,1.00E-06,0.000107485260618,1.00E-06,1.00E-06,1.52167800668037E-05,0.000721185331166,6.08004789120523E-05,0.000561307787544,2.25903683039772E-05,1.50207156135449E-05,1.00E-06,0.001393079169983,0.001079690824854,1.00E-06,5.4662827803081E-05,0.000910753943082,1.00E-06,0.00068480347357,0.001227366055884,0.000263443029256])
mustr=mustr[0:111]
mustrunc=mustr*0.5 # *** Mutation rate uncertainties - could replace with realistic uncertainties
mustrunc+=5.e-7
ygen=33 # Years per generation
ygenunc=2 # Years per generation uncertainty
mustr/=ygen # Convert mutation rates to years
mustrunc/=ygen*(1+ygenunc/ygen)
nstr=len(mustr) # Get maximum number of STRs to consider
# Mutation direction probabilities (positive/negative change in STR allele)
wp=0.5 # Equal probability of positive and negative mutations
wn=wp  # *** wn!=wp not coded here!
# STR multi-step probabilities
omegap=np.array([1,0.9621685978,0.032,0.004,0.0012649111,0.0004,0.000124649111,0.00004]) # Estimated from Ballantyne et al. (2010)
omegap/=2
omegan=omegap
# SNP mutation rate and uncertainty
musnp=8.330e-10 #7.547e-10 # SNP creation rate /base pair/year
musnpunc=0.800e-10 #0.661e-10
muindel=5.75e-11 # Indel rate (estimated from 315 indels versus 4137 SNPs in Build 37 report)
muindelunc=0.504e-11 # Based on quadrature sum of sqrt(315)/4137 and musnpunc/musnp
maxsnps=100 # Maximum number of mutations to consider when forming tree
snpstep=0.01 # Step size in mutation timescales
ylength=60000000 # Maximum number of base pairs to consider

# Generate timeline
maxtime=8000 # years into the past
timestep=1 #years
times=np.arange(0,maxtime,timestep)
nowtime=2020 # present day (AD/CE)
dates=np.arange(nowtime,nowtime-maxtime,-timestep)

# Set up parameters for generating the PDFs for STRs
maxg=8 # Max GD considered < 5 = quick, but probably normally want ~10 for a full tree
maxm=5 # Max number of mutations considered < close relationships ~5, probably normally want ~20 for a large haplogroup
maxnm=20 # Max number of mutation timescales to compute maxm mutations to < probably 30 if CDY fast
maxq=len(omegap) # Max GD caused by a multi-step mutation
maxnq=3 # Max number of multi-step mutations to consider in one lineage < 3 maybe ok but might want ~4
mres=0.01 # Resolution of initial probability calculation (mutation timescales) < may want finer timescale, maybe ok
mtimes=np.arange(0,maxnm,mres)
pdf_str_m=np.zeros((maxg,len(mtimes))) # Probability density function of obtaining genetic distance (g) on any given STR in mutation timescale (m)
pdf_str_t=np.zeros((len(mustr),maxg,len(times))) # Probability density function of obtaining genetic distance (g) on specific STRs in physical timescale (t)

# Set up default date of TMRCA
dt=dt0/sqrt(2) # Average birth date uncertainty is reduced by factor sqrt(2) for two testers
st=sqrt(st0**2+dt**2) # Add uncertainty in the mean
tmrca=np.zeros(len(dates))
tmrca[0]=1
dmrca0=ages2dates(tmrca,dates,times,nowtime,t0,st) # Convert TMRCA to dates
np.savetxt("dmrca0.dat", np.c_[dates,dmrca0], delimiter=' ')


print ("Genetic distance setup")
# Generate the generic PDF in terms of mutation timescales (p_g(\bar{m}_s))
generate_genpdfstr(mtimes,pdf_str_m,maxg,maxm,maxq,maxnq,maxnm,mres,omegap,omegan)
# Generate the specific PDFs for each STR
generate_pdfstr(mtimes,times,pdf_str_m,pdf_str_t,mustr,mustrunc,maxtime,maxg)







# ===================================================
# Main code - run example 4 in McDonald et al. (2021)
# ===================================================
# Include paper trail, STRs only, SNPs only, indels
default_do_strs = 1
default_do_snps = 1
default_do_indels = 0
default_do_paper = 1

# Ancestral haplotype for R-S781
ht0=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])



# R-A889 [315522,349908]
# This is a simple calculation with two testers
print ("--- R-A889 ---")
# Paper trail information
testpaper1=1774 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-down"
testpaper2=1774 # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-down"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.39e6 # Shared coverage of both tests
# SNP results
snp_da=0 # Private SNPs in first test
snp_db=3 # Private SNPs in second test
# STR results
ht1=np.array([13,24,14,11,11,14,12,12,11,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht2=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_a889=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

dmrca_a889,dmrca1,dmrca2,tmrca_a889,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_a889,ht1,ht2,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_a889)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_a889.dat", np.c_[dates,dmrca_a889], delimiter=' ')



# R-A921 [187778,30331]
# Another simple calculation with two testers
print ("--- R-A921 ---")
# Paper trail information
testpaper1=1855 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=1855 # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.59e6 # Shared coverage of both tests
# SNP results
snp_da=0 # Private SNPs in first test
snp_db=1 # Private SNPs in second test
# STR results
ht1=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,36,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,37,15,9,16,12,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht2=np.array([13,24,14,11,11,14,12,12,12,13,12,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,37,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,20,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_a921=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

dmrca_a921,dmrca1,dmrca2,tmrca_a921,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_a921,ht1,ht2,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_a921)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_a921.dat", np.c_[dates,dmrca_a921], delimiter=' ')



# R-A922 [R-A921, RDNDZ]
# No STR results are publicly available for RDNDZ, so we can't get a clear STR motif for this intermediate clade
# No constraint on the paper trail exists either, so this isn't performed
# Because this is not just two testers, accommodation has to be made to add on the age of R-A921 after the TMRCA calculation
print ("--- R-A922 ---")
# Paper trail information
testpaper1=nowtime # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=nowtime # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.59e6 # Shared coverage of both tests
# SNP results
snp_da=3 # Private SNPs in first test
snp_db=2 # Private SNPs in second test
# STR results
#ht1=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,36,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,37,15,9,16,12,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
#ht2=np.array([13,24,14,11,11,14,12,12,12,13,12,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,37,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,20,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
#hta=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests

# Which calculations can be performed
do_strs=0
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

dmrca_a922,dmrca1,dmrca2,tmrca_a922,tmrca1,tmrca2=test2(dmrca_a921,dmrca0,ht0,ht0,ht0,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_a922)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_a922.dat", np.c_[dates,dmrca_a922], delimiter=' ')



# R-FT211732 [183344,181994]
# Another simple calculation with two testers
print ("--- R-FT211732 ---")
# Paper trail information
testpaper1=1765 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-down"
testpaper2=1765 # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-down"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=13.81e6 # Shared coverage of both tests
# SNP results
snp_da=1 # Private SNPs in first test
snp_db=2 # Private SNPs in second test
# STR results
ht1=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,20,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht2=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,24,12,23,18,10,14,17,9,12,11])
ht_ft211732=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

dmrca_ft211732,dmrca1,dmrca2,tmrca_a922,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_ft211732,ht1,ht2,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_a922)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_ft211732.dat", np.c_[dates,dmrca_ft211732], delimiter=' ')




# R-FGC74572 [R-A889,R-A922,R-FT211732,446440,292186,B136219,48694]
print ("--- R-FGC74572 ---")
# R-A922 and B136219 don't have STRs
# Since the coverage changes significant each time, perform the calculation separately for all clades
# Paper trail information - valid for all since all pre-existing data has been entered
testpaper1=nowtime # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=nowtime # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.39e6 # Shared coverage of both tests
# SNP results
snp_a889=1 # Private SNPs
snp_a922=2 # Private SNPs
snp_ft211732=1 # Private SNPs
snp_446440=1 # Private SNPs
snp_292186=2 # Private SNPs
snp_B136219=2 # Private SNPs
snp_48694=4 # Private SNPs
# STR results
ht_446440=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,10,10,19,23,15,15,18,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,11,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,24,12,23,18,10,14,17,9,12,11])
ht_292186=np.array([13,24,14,11,11,14,12,12,12,13,13,29,16,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,38,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,17,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,22,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_48694=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,36,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,23,21,14,12,11,13,11,11,12,11,35,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,13,14,24,14,10,10,20,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_fgc74572=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht_a889,ht_ft211732,ht_446440,ht_292186,ht_48694))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests

# R-A889
do_strs=default_do_strs
coverage=9.39e6
dmrca,dmrca_fgc74572_1,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca_a889,dmrca0,ht_fgc74572,ht_a889,ht_fgc74572,snp_a889,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
# R-A922
do_strs=0
coverage=9.72e6
dmrca,dmrca_fgc74572_2,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca_a922,dmrca0,ht_fgc74572,ht_fgc74572,ht_fgc74572,snp_a922,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
# R-FT211732
do_strs=default_do_strs
coverage=13.81e6
dmrca,dmrca_fgc74572_3,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca_ft211732,dmrca0,ht_fgc74572,ht_ft211732,ht_fgc74572,snp_ft211732,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
# 446440
do_strs=default_do_strs
coverage=9.43e6
dmrca,dmrca_fgc74572_4,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_fgc74572,ht_446440,ht_fgc74572,snp_446440,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
# 292186
do_strs=default_do_strs
coverage=8.91e6
dmrca,dmrca_fgc74572_5,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_fgc74572,ht_292186,ht_fgc74572,snp_292186,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
# B136219
do_strs=0
coverage=8.32e6
dmrca,dmrca_fgc74572_6,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_fgc74572,ht_fgc74572,ht_fgc74572,snp_B136219,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
# 48694
do_strs=default_do_strs
coverage=9.41e6
dmrca,dmrca_fgc74572_7,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_fgc74572,ht_48694,ht_fgc74572,snp_48694,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)

dmrca_fgc74572=dmrca_fgc74572_1*dmrca_fgc74572_2*dmrca_fgc74572_3*dmrca_fgc74572_4*dmrca_fgc74572_5*dmrca_fgc74572_6*dmrca_fgc74572_7
dmrca_fgc74572/=np.sum(dmrca_fgc74572)
cumtmrca=np.cumsum(dmrca_fgc74572)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
#np.savetxt("dmrca_fgc74572.dat", np.c_[dates,dmrca_fgc74572], delimiter=' ')
np.savetxt("dmrca_fgc74572.dat", np.c_[dates,dmrca_fgc74572,dmrca_fgc74572_1,dmrca_fgc74572_2,dmrca_fgc74572_3,dmrca_fgc74572_4,dmrca_fgc74572_5,dmrca_fgc74572_6,dmrca_fgc74572_7], delimiter=' ')



# R-A309 [E16164,E15052] (this is the Corsican branch)
# This is a simple calculation with two testers
print ("--- R-A309 ---")
# Paper trail information
testpaper1=1735 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=1735 # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.35e6 # Shared coverage of both tests
# SNP results
snp_da=0 # Private SNPs in first test
snp_db=7 # Private SNPs in second test
# STR results
ht1=np.array([13,24,14,11,11,14,12,12,13,13,13,29,17,9,10,11,11,25,15,19,29,14,15,16,17,11,10,19,23,15,15,17,18,36,36,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,16,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht2=np.array([13,25,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,16,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,16,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_a309=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

dmrca_a309,dmrca1,dmrca2,tmrca_a309,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_a309,ht1,ht2,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_a309)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_a309.dat", np.c_[dates,dmrca_a309], delimiter=' ')



# R-BY39565 [523840,888640]
# This is a simple calculation with two testers
print ("--- R-BY39565 ---")
# Paper trail information
testpaper1=1740 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=1740 # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.58e6 # Shared coverage of both tests
# SNP results
snp_da=2 # Private SNPs in first test
snp_db=3 # Private SNPs in second test
# STR results
ht1=np.array([13,25,14,10,11,14,12,12,12,13,13,30,18,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,16,15,17,17,36,38,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,24,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,20,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht2=np.array([13,25,15,10,11,14,12,12,12,13,13,30,18,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,16,15,17,17,36,38,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,24,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_by39565=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht1,ht2))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

dmrca_by39565,dmrca1,dmrca2,tmrca_by39565,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_by39565,ht1,ht2,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_by39565)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_by39565.dat", np.c_[dates,dmrca_by39565], delimiter=' ')



# R-unnamed [R-BY39565,128499,500420]
# This clade is defined by two common STR mutations, but no SNPs
# The coverage of these three clades is very similar, so we can compute the upstream part from R-BY39565 first and individually
# Then add on the combination of the other two testers
print ("--- R-unnamed ---")
# Paper trail information
testpaper1=1787 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=nowtime # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.58e6 # Shared coverage of both tests
# SNP results
snp_by39565=4
snp_128499=2
snp_500420=6
# STR results
ht_128499=np.array([13,24,14,11,11,14,12,12,12,13,13,30,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,16,15,17,17,36,38,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,35,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,13,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_500420=np.array([13,24,14,11,11,14,12,12,12,13,13,30,17,9,9,11,11,25,15,19,29,14,15,17,17,11,10,19,23,16,15,16,17,36,38,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,12,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,12,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_unnamed=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht_by39565,ht_128499,ht_500420))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

# Note the difference in position for dumrca_unnamed_*
dmrca,dmrca_unnamed_1,dmrca2,tmrca,tmrca_unnamed_1,tmrca2=test2(dmrca_by39565,dmrca0,ht_unnamed,ht_by39565,ht_unnamed,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_unnamed_2,dmrca1,dmrca2,tmrca_unnamed_2,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_unnamed,ht_128499,ht_500420,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_unnamed=dmrca_unnamed_1*dmrca_unnamed_2
dmrca_unnamed/=np.sum(dmrca_unnamed)

cumtmrca=np.cumsum(dmrca_unnamed)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_unnamed.dat", np.c_[dates,dmrca_unnamed], delimiter=' ')



# R-17651002 [R-unnamed,737640,860987]
# This clade is defined by an unnamed mutation on palindromic arm 5 at GRCH38 position 17651002 (G->A) (elsewhere called A22207).
# www.s781.org gives this as the Lorn-Appin branch
# The coverage of these three clades is very similar, so we can progress in the same way as the previous clade: the STR-defined clade first,
# Then add on the combination of the other two testers
print ("--- R-17651002 ---")
# Paper trail information
testpaper1=1787 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=nowtime # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.65e6 # Shared coverage of both tests
# SNP results
snp_unnamed=0
snp_737640=2
snp_860987=6
# STR results
ht_737640=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,38,13,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_860987=np.array([13,24,14,11,11,14,12,12,12,13,12,29,17,9,9,11,11,24,15,19,29,14,14,17,17,11,10,19,23,15,15,17,17,36,36,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,15,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_17651002=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht_unnamed,ht_737640,ht_860987))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

# Note the difference in position for dumrca_unnamed_*
dmrca,dmrca_17651002_1,dmrca2,tmrca,tmrca_17651002_1,tmrca2=test2(dmrca_unnamed,dmrca0,ht_17651002,ht_unnamed,ht_unnamed,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_17651002_2,dmrca1,dmrca2,tmrca_17651002_2,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_17651002,ht_737640,ht_860987,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_17651002=dmrca_17651002_1*dmrca_17651002_2
dmrca_17651002/=np.sum(dmrca_17651002)

cumtmrca=np.cumsum(dmrca_17651002)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_17651002.dat", np.c_[dates,dmrca_17651002], delimiter=' ')



# R-A5020 [235915,520340]
# This is a simple calculation with two testers
# There are no public STR results for the second tester (520340)
# Hence we cannot calculate a STR-based TMRCA here
# 235915 has one STR mutation from the ancestral motif
# Given the disparity in the upstream (5) and downstream (1,1) SNP mutations,
# it is likely that the STR mutation is upstream of R-A5020, but we can't say this for certain.
# No public ancestry information exists either
print ("--- R-A5020 ---")
# Paper trail information
testpaper1=nowtime # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=nowtime # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=8.34e6 # Shared coverage of both tests
# SNP results
snp_da=1 # Private SNPs in first test
snp_db=1 # Private SNPs in second test
# STR results
ht_a5020=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,12,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
# Which calculations can be performed
do_strs=0
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=0

dmrca_a5020,dmrca1,dmrca2,tmrca_a5020,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht0,ht0,ht0,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_a5020)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_a5020.dat", np.c_[dates,dmrca_a5020], delimiter=' ')



# R-A5025 [115205,B59459,B517463]
# This is a relatively simple calculation with three testers
print ("--- R-A5025 ---")
# Paper trail information
testpaper1=1792 # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-up"
testpaper2=1792 # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-up"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=9.52e6 # Shared coverage of both tests
# SNP results
snp_115205=1 # Private SNPs in first test
snp_B59459=3 # Private SNPs in second test
snp_B517463=2 # Private SNPs in second test
# STR results
ht_115205=np.array([13,24,14,11,11,13,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,15,15,16,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,14,11,11,12,11,36,15,9,16,13,26,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,20,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_B59459=np.array([13,24,14,11,11,14,12,13,12,13,13,29,17,9,10,11,11,26,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,14,11,11,12,11,36,15,9,16,13,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_B517463=np.array([13,24,14,12,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,15,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,16,8,12,22,21,14,12,11,14,11,11,12,11,36,15,9,16,12,25,26,19,12,11,11,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,20,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
ht_a5025=np.ndarray.flatten(scipy.stats.mode(np.array((ht0,ht_115205,ht_B59459,ht_B517463))).mode) # Compute ancestral haplotype as modal of global ancestral haplotype and both tests
# Which calculations can be performed
do_strs=default_do_strs
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

# Note difference in placement of _1 and _2
dmrca_a5025_1,dmrca1,dmrca2,tmrca_a5025_1,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht_a5025,ht_115205,ht_B59459,snp_115205,snp_B59459,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca,dmrca_a5025_2,dmrca2,tmrca,tmrca_a5025_2,tmrca2=test2(dmrca0,dmrca0,ht_a5025,ht_B517463,ht0,snp_B517463,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_a5025=dmrca_a5025_1*dmrca_a5025_2
dmrca_a5025/=np.sum(dmrca_a5025)

cumtmrca=np.cumsum(dmrca_a5025)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_a5025.dat", np.c_[dates,dmrca_a5025], delimiter=' ')



# R-A5021 [143035,783534]
# This is a simple calculation with two testers
# There are no public STR results
# No public ancestry information exists either
print ("--- R-A5021 ---")
# Paper trail information
testpaper1=nowtime # Paper trail info for line 1
testpaperunc1=1
testpaperalpha1=1
testpapertype1="step-down"
testpaper2=nowtime # Paper trail info for line 2
testpaperunc2=1
testpaperalpha2=1
testpapertype2="step-down"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
coverage=8.90e6 # Shared coverage of both tests
# SNP results
snp_da=4 # Private SNPs in first test
snp_db=3 # Private SNPs in second test
# STR results
#ht_a5020=np.array([13,24,14,11,11,14,12,12,12,13,13,29,17,9,10,11,11,25,15,19,29,14,15,17,17,11,10,19,23,15,15,17,17,36,37,12,12,11,9,15,16,8,10,10,8,11,10,12,23,23,16,10,12,12,15,8,12,22,21,14,12,11,13,11,11,12,11,36,15,9,16,12,25,26,19,12,11,12,12,11,9,13,12,10,11,11,30,12,14,24,13,10,10,21,15,18,13,24,16,12,15,25,12,23,18,10,14,17,9,12,11])
# Which calculations can be performed
do_strs=0
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=0

dmrca_a5021,dmrca1,dmrca2,tmrca_a5021,tmrca1,tmrca2=test2(dmrca0,dmrca0,ht0,ht0,ht0,snp_da,snp_db,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
cumtmrca=np.cumsum(dmrca_a5021)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))
np.savetxt("dmrca_a5021.dat", np.c_[dates,dmrca_a5021], delimiter=' ')



# R-S781 [R-FGC74572,R-A309,R-17651002,R-A5020,R-A5025,R-A5021]
# Final calculation
# Due to the different coverage (only R-FGC74572 has two tests with ~13 Mbp, the rest have ~9 Mbp),
# we'll do R-FGC74572 separately, then the rest pair-wise
# We'll save the date-of-MRCA (dmrca) for each clade, because we'll use these again later.
print ("--- R-S781 ---")
# Paper trail information
testpaper1=1245 # Paper trail info for line 1
testpaperunc1=20
testpaperalpha1=1
testpapertype1="smooth"
testpaper2=1245 # Paper trail info for line 2
testpaperunc2=20
testpaperalpha2=1
testpapertype2="smooth"
testpapers=-9999 # Surname restriction
testpaperuncs=20
testpapertypes="smooth-end" # -end for shared surname, -start for not shared
# Coverage
# SNP results
snp_fgc74572=2
snp_a309=4
snp_17651002=1
snp_a5020=5
snp_a5025=3
snp_a5021=4
# STR results - these have all been computed

# Which calculations can be performed
do_snps=default_do_snps
do_indels=default_do_indels
do_paper=default_do_paper

# FGC74572 (high coverage)
do_strs=default_do_strs
coverage=13.40e6
dmrca,dmrca1,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca_fgc74572,dmrca0,ht0,ht_fgc74572,ht0,snp_fgc74572,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_s781_fgc74572=dmrca1
tmrca_s781_fgc74572=tmrca1

# 17651002, A309
do_strs=default_do_strs
coverage=9.50e6 # Shared coverage of both tests
dmrca,dmrca1,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca_a309,dmrca_17651002,ht0,ht_a309,ht_17651002,snp_a309,snp_17651002,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_s781_17651002=dmrca1
dmrca_s781_a309=dmrca2
tmrca_s781_17651002=tmrca1
tmrca_s781_a309=tmrca2

# A5020, A5021 (no STRs)
do_strs=0
coverage=8.62e6 # Shared coverage of both tests
dmrca,dmrca1,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca_a5020,dmrca_a5021,ht0,ht0,ht0,snp_a5020,snp_a5021,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_s781_a5020=dmrca1
dmrca_s781_a5021=dmrca2
tmrca_s781_a5020=tmrca1
tmrca_s781_a5021=tmrca2

# A5025
coverage=9.52e6
dmrca,dmrca,dmrca2,tmrca,tmrca1,tmrca2=test2(dmrca_a5025,dmrca0,ht0,ht_a5025,ht0,snp_a5025,0,coverage,testpaper1,testpaperunc1,testpaperalpha1,testpapertype1,testpaper2,testpaperunc2,testpaperalpha2,testpapertype2,testpapers,testpaperuncs,testpapertypes,do_strs,do_snps,do_indels,do_paper)
dmrca_s781_a5025=dmrca1
tmrca_s781_a5025=tmrca1

dmrca_s781=dmrca_s781_fgc74572*dmrca_s781_17651002*dmrca_s781_a309*dmrca_s781_a5020*dmrca_s781_a5021*dmrca_s781_a5025
dmrca_s781/=np.sum(dmrca_s781)

dmrca_s781_known=dmrca_s781_17651002*dmrca_s781_a309
dmrca_s781_known/=np.sum(dmrca_s781_known)

cumtmrca=np.cumsum(dmrca_s781)
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))

cumtmrca=np.cumsum(dmrca_s781_known)
print ('Only known lines')
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 68.3% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.1585])],dates[len(cumtmrca[cumtmrca<0.8415])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
print ('Combined 99.5% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.0025])],dates[len(cumtmrca[cumtmrca<0.9975])]))


np.savetxt("dmrca_s781.dat", np.c_[dates,dmrca_s781], delimiter=' ')
np.savetxt("dmrca_s781_fgc74572.dat", np.c_[dates,dmrca_s781_fgc74572], delimiter=' ')
np.savetxt("dmrca_s781_17651002.dat", np.c_[dates,dmrca_s781_17651002], delimiter=' ')
np.savetxt("dmrca_s781_a309.dat", np.c_[dates,dmrca_s781_a309], delimiter=' ')
np.savetxt("dmrca_s781_a5020.dat", np.c_[dates,dmrca_s781_a5020], delimiter=' ')
np.savetxt("dmrca_s781_a5021.dat", np.c_[dates,dmrca_s781_a5021], delimiter=' ')
np.savetxt("dmrca_s781_a5025.dat", np.c_[dates,dmrca_s781_a5025], delimiter=' ')



# Now let's constrain things on the way down.
# We'll only do this for the first set of haplogroups, but there's no reason we couldn't apply this further down the tree too
dmrca2_fgc74572=np.convolve(dmrca_s781,tmrca_s781_fgc74572,mode='full')[-len(tmrca1):]*dmrca_fgc74572
dmrca2_17651002=np.convolve(dmrca_s781,tmrca_s781_17651002,mode='full')[-len(tmrca1):]*dmrca_17651002
dmrca2_a309=np.convolve(dmrca_s781,tmrca_s781_a309,mode='full')[-len(tmrca1):]*dmrca_a309
dmrca2_a5020=np.convolve(dmrca_s781,tmrca_s781_a5020,mode='full')[-len(tmrca1):]*dmrca_a5020
dmrca2_a5021=np.convolve(dmrca_s781,tmrca_s781_a5021,mode='full')[-len(tmrca1):]*dmrca_a5021
dmrca2_a5025=np.convolve(dmrca_s781,tmrca_s781_a5025,mode='full')[-len(tmrca1):]*dmrca_a5025

dmrca2_fgc74572/=np.sum(dmrca2_fgc74572)
dmrca2_17651002/=np.sum(dmrca2_17651002)
dmrca2_a309/=np.sum(dmrca2_a309)
dmrca2_a5020/=np.sum(dmrca2_a5020)
dmrca2_a5021/=np.sum(dmrca2_a5021)
dmrca2_a5025/=np.sum(dmrca2_a5025)

cumtmrca=np.cumsum(dmrca2_fgc74572)
print ("--- R-FGC74572 ---")
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
cumtmrca=np.cumsum(dmrca2_17651002)
print ("--- R-17651002 ---")
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
cumtmrca=np.cumsum(dmrca2_a309)
print ("--- R-A309 ---")
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
cumtmrca=np.cumsum(dmrca2_a5020)
print ("--- R-A5020 ---")
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
cumtmrca=np.cumsum(dmrca2_a5021)
print ("--- R-A5021 ---")
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))
cumtmrca=np.cumsum(dmrca2_a5025)
print ("--- R-A5025 ---")
print ('Combined Central estimate:          {} CE'.format(dates[len(cumtmrca[cumtmrca<0.5])]))
print ('Combined 95.0% confidence interval: {} -- {} CE'.format(dates[len(cumtmrca[cumtmrca<0.025])],dates[len(cumtmrca[cumtmrca<0.975])]))

np.savetxt("dmrca2_fgc74572.dat", np.c_[dates,dmrca2_fgc74572], delimiter=' ')
np.savetxt("dmrca2_17651002.dat", np.c_[dates,dmrca2_17651002], delimiter=' ')
np.savetxt("dmrca2_a309.dat", np.c_[dates,dmrca2_a309], delimiter=' ')
np.savetxt("dmrca2_a5020.dat", np.c_[dates,dmrca2_a5020], delimiter=' ')
np.savetxt("dmrca2_a5021.dat", np.c_[dates,dmrca2_a5021], delimiter=' ')
np.savetxt("dmrca2_a5025.dat", np.c_[dates,dmrca2_a5025], delimiter=' ')
