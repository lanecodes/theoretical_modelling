# Module makes use of class composition rather than inheritance

from datetime import datetime
import numpy as np
import zipfile, h5py

class Experiment():
    def __init__(self,dbase_dir,params,Tdict,jamp_dict,No_Jruns,save_ival):
        self.dbase_dir=dbase_dir
        
        # params should be a dict of the form:
        # p={'N' : n, 'R' : replicas, 'x0': x0amp, 't' : time, 'dt' : delta, 
        #    'eta':eta}
        self.p=params
        
        # Tdict of the form {'Tmin': Tmin, 'Tmax': Tmax, 'Tint':Tinterval}
        self.Tdict = Tdict
        
        # jamp_dict of the form {'jampmin': jamp_min, 'jampmax': jamp_max, 
        # 'jampint':jamp_interval}
        self.jamp_dict = jamp_dict
        
        self.Jruns = No_Jruns #number of interaction matrices to average over
        
        self.save_ival=save_ival #no time-steps between writes to disk, used 
                                 #for calc sim time

        self.Tarray = np.arange(Tdict['Tmin'],
                                Tdict['Tmax']+Tdict['Tint'],Tdict['Tint'])
        self.jamparray = np.arange(jamp_dict['jampmin'],
                                   jamp_dict['jampmax']+jamp_dict['jampint'],
                                   jamp_dict['jampint'])

        self.quant_getters = {'mean_error' : 'getMeanDeviation',
                              'variance_error' : 'getVarDeviation',
                              'SEM' : 'getSEM'
                              }
                              
        self.times = [i * save_ival * self.p['dt'] for i in 
                      range(int(self.p['t']/(self.p['dt'] * save_ival))+1)]
        
        self.name=('N'+str(params['N'])+'-R'+str(params['R'])+
                         '-x'+str(int(params['x0']))+'-t'+str(params['t'])+
                         '-dt'+str(params['dt'])+'-eta'+
                         str(self.intCheck(params['eta']))
                   )

        self.s_file_name = self.dbase_dir +'/'+self.name +'/'+'AveOverJ.hdf5'

        # list of 'Run' objects (which themselves contain simulations)
        self.runs=[Run(self.dbase_dir,self.name, self.p, self.Tarray, 
                       self.jamparray ,self.times,i+1) for i in range(self.Jruns)]

        
    def intCheck(self,num):
        if int(num) == num:
            return int(num)
        else:
            return num    

    def getTindex(self,T):
        # given a T value, get its position in simulation array
        return np.where(self.Tarray==T)[0][0]

    def getjampindex(self,jamp):
        # given a T value, get its position in simulation array
        return np.where(self.jamparray==jamp)[0][0]

    def calcAveOverJ(self,quantity,method,verbose):
        if quantity in self.quant_getters.keys():
            f_out = h5py.File(self.s_file_name,'a')

            try:
                f_means = f_out.create_group(quantity)
            except ValueError:
                f_means=f_out[quantity]

            try: 
                f_meth = f_means.create_group(method)
            except ValueError:
                f_meth=f_means[method]

            for i in range(len(self.Tarray)):
                for j in range(len(self.jamparray)):
                    fname = ('T'+str(self.intCheck(self.Tarray[i]))+
                             '-J'+str(self.intCheck(self.jamparray[j]))
                             )
                
                    if verbose:
                        print 'processing ' + fname

                    # use getattr to return method for obtaining e.g. means/ variances
                    # from appropriate simulation objects in loop, then average over
                    # each array for each J run.
                    means = np.array([getattr(self.runs[run].simulations[i][j],
                                              self.quant_getters[quantity])(method)
                                      for run in range(self.Jruns)])

                    ave_on_J = np.mean(means,axis=0,dtype='float32')

                    try:
                        f_meth.create_dataset(name=fname,data=ave_on_J,
                                              compression='gzip')
                    except RuntimeError:
                        del f_meth[fname]
                        f_meth.create_dataset(name=fname,data=ave_on_J,
                                              compression='gzip')

            f_out.close()
            print 'calculation of ' + quantity + ' for ' + method + ' complete.'
        else:
            print 'not a recognised quantity!'

    def getAveOverJ(self,quantity,method,verbose):
        if quantity in self.quant_getters.keys():
            f_in = h5py.File(self.s_file_name,'a')

            try:
                f_means = f_in[quantity+'/'+method]
            except ValueError:
                print 'could not find data'

            dat = []

            for i in range(len(self.Tarray)):
                Trow=[]
                for j in range(len(self.jamparray)):
                    fname = ('T'+str(self.intCheck(self.Tarray[i]))+
                             '-J'+str(self.intCheck(self.jamparray[j]))
                             )
                    if verbose:
                        print 'processing ' + fname

                    Trow.append(f_means[fname][...])
                dat.append(Trow)

            f_in.close()
            return dat
        else:
            print 'not a recognised quantity!' 

    def __str__(self):
        return self.name

class Run():
        def __init__(self,dbase_dir,exp_name,params,Tarray,jamparray,times,Jrun):
            self.dbase_dir=dbase_dir
            #string representation of experiment name for finding files
            self.exp_name=exp_name 
            self.p = params
            self.Jrun=Jrun
            self.run_direc=(self.dbase_dir+'/'+self.exp_name+'/Jrun'+
                            str(Jrun)+'/'
                            )
            
            self.Tarray = Tarray
            self.jamparray = jamparray
            self.times = times

            self.simulations=[]
            for T in self.Tarray:
                Trow=[]
                for jamp in self.jamparray:
                    Trow.append(Simulation(self.run_direc,self.p,T,times,jamp))
                self.simulations.append(Trow)  

class Simulation():
    def __init__(self,zfile_direc,params,T,times,jamp):
        self.p=params
        self.varp_str='T'+str(self.intCheck(T))+'-J'+str(self.intCheck(jamp))
        self.times = times
        self.zfile=zfile_direc+self.varp_str+'.zip'
        
    def intCheck(self,num):
        if int(num) == num:
            return int(num)
        else:
            return num
            
    def getMean(self,method):
        with zipfile.ZipFile(self.zfile, 'r') as myzip:
            mfile = myzip.open(self.varp_str+'.'+method+'M','r')
            mu=np.genfromtxt(mfile,delimiter=',')
        return mu

    def getVar(self,method):
        with zipfile.ZipFile(self.zfile, 'r') as myzip:
            mfile = myzip.open(self.varp_str+'.'+method+'V','r')
            var=np.genfromtxt(mfile,delimiter=',')
        return var
    
    def get_J(self):
        with zipfile.ZipFile(self.zfile, 'r') as myzip:
            mfile = myzip.open(self.varp_str+'.J','r')
            J=np.genfromtxt(mfile,delimiter=',')
        return J

    def get_x0(self):
        with zipfile.ZipFile(self.zfile, 'r') as myzip:
            mfile = myzip.open(self.varp_str+'.DM','r')
            lines = mfile.readlines()
            x0=np.fromiter((line.split(',')[0] for line in lines), np.float)
        return x0
    
    def getMeanDeviation(self,method):
        return np.abs(self.getMean('D')-self.getMean(method))
    
    def getVarDeviation(self,method):
        true_var = self.getVar('D')
        approx_var = self.getVar(method)
        return np.abs(approx_var-true_var)/true_var

    def getSEM(self,method):
        # see expression for standard error of the mean
        return np.sqrt(self.getVar('D')/self.p['R'])

    def getConnCorrelator(self,method):
        with zipfile.ZipFile(self.zfile, 'r') as myzip:
            mfile = myzip.open(self.varp_str+'.'+method+'I','r')
            CC=np.genfromtxt(mfile,delimiter=',')
        return CC

    def getRunDur(self,method):
        with zipfile.ZipFile(self.zfile, 'r') as myzip:
            mfile = myzip.open(self.varp_str+'.meta','r')
            lines = [line.rstrip('\n') for line in mfile.readlines()]
            for line in lines:
                if line[0] == method:
                    t = np.float(line.split(':')[1])
                else:
                    pass
        return t

    def getRunTime(self):
        with zipfile.ZipFile(self.zfile, 'r') as myzip:
            mfile = myzip.open(self.varp_str+'.meta','r')
            pts = mfile.readline().rstrip('\n').split(':')[1:]
            date = pts[0]+':'+pts[1]+':'+pts[2]
            dtime = datetime.strptime(date,' %d-%b-%Y %H:%M:%S' )
        return dtime
    
    def __str__(self):
        return self.varp_str
