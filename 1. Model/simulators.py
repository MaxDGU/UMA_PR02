# -*- coding: utf-8 -*-

from uma import *
from models import *

import re, json
from datetime import datetime
from itertools import product

class ProblemSet:
    """ A ProblemSet is a set of arithmetic problems with convenient functions for sampling from the problems, running a model on them, and displaying output in a readable format. """
    def __init__(self, prob_set=None, filter=None, N=None):
        self.P = ProblemSet.load(prob_set)
        if filter is not None:
            self.filter(filter)
        if N is not None:
            self.sample(N)
    def __str__(self):
        return(str(self.P))
    @classmethod
    def load(cls, prob_set):
        test_sets   = ['sd_add', 'sd_mul', 'sp2013', 'bss2021']
        train_sets  = ['go_math', 'go_math_fractions', 'go_math_decimals', 'go_math_rational']
        if prob_set is None:
            probs = []
        elif type(prob_set) is list:
            probs = prob_set
        elif prob_set in test_sets:
            F = 'Problem Sets Testing/' + prob_set + '.csv'
            probs = pd.read_csv(F)['prob'].tolist()
        elif prob_set in train_sets:
            F = 'Problem Sets Training/' + prob_set + '.csv'
            probs = pd.read_csv(F)['prob'].tolist()
        else:
            print("Unrecognized value for prob_set in Simulator.getProblemSet: %s" % str(prob_set))
        P = pd.DataFrame([ProblemSet.probToDict(prob) for prob in probs])
        P.index = range(P.shape[0])
        return(P)
    @classmethod
    def probToDict(cls, prob):
        """
        Convert a problem to a {prob, notation, operation, subtype, key} dict
        Uses similar (not identical) values to feature dimensions used by Contexts
        This is a convenience function to be used by simulations, not by UMA itself
        """
        C           = Context(prob)
        features    = C.getFeatures()
        notation    = list(set(number_types).intersection(features))[0]
        operation   = {
            '+': 'add',
            '-': 'sub',
            '*': 'mul',
            ':': 'div'}[list(set(operations).intersection(features))[0]]
        try:
            operands = list(set(['SD', 'MD', 'FR', 'MI', 'WF', 'WD', 'DD']).intersection(features))[0]
            L = list(set(['ED', 'UD', 'EDD', 'UDD']).intersection(features))
            denoms = L[0] if len(L)>0 else ""
        except:
            operands = ""
            denoms = ""
        return({
            'prob':         str(prob),
            'notation':     notation, 
            'operation':    operation, 
            'operands':     operands,
            'denoms':       denoms,
            'key':          myEval(prob)})
    def filter(self, filter):
        Q = self.P.copy()
        if type(filter) is str:
            filter = [filter]
        if "rational" in filter:
            Q = Q[Q['notation'].isin(['fraction', 'decimal'])]
        if "base_ten" in filter:
            Q = Q[Q['notation'].isin(['SD_whole', 'MD_whole', 'decimal'])]
        if "multidigit" in filter:
            Q = Q[(Q['notation']=='MD_whole')|(Q['notation']=='decimal')]
        for var in ['notation', 'operation', 'operands']:
            for tc in set(Q[var]):
                if tc in filter:
                    Q = Q[Q[var]==tc]
        Q.index = range(Q.shape[0])
        self.P = Q
    def sample(self, N):
        Q = self.P.copy()
        if N<Q.shape[0]:
            Q = Q.sample(N, axis=0)
        Q.index = range(Q.shape[0])
        self.P = Q
    def runModel(self, M, learn=False, verbose=False, force_rules=None, record_rules=None):
        D           = self.P.copy()
        D           = D.reindex(columns = D.columns.tolist() + ['steps', 'time', 'acc', 'ans'])
        D['ans']    = ""
        if record_rules is not None:
            record_names = [r.name for r in record_rules]
            for name in record_names:
                D[name] = 0
        self.doOnStart(M)
        for i in range(D.shape[0]):
            if verbose:
                print("\rProblem %s of %s    " % (i+1, D.shape[0]), end="")
            self.doOnIter(M,i)
            p = D.index[i] # just in case ith index of D isn't the same as i itself
            M.run(State(D.loc[p,'prob']), learn=learn, verbose=False, force_rules=force_rules)
            D.loc[p,'steps']    = len(M.trace)
            D.loc[p,'time']     = M.runtime
            answer              = M.trace[-1][0].ws[0].features['answer']
            try:
                x               = float(answer)
                D.loc[p,'acc']  = 0+(round(x,6)==round(D.loc[p,'key'],6))
                D.at[p,'ans']   = str(answer) if answer.features['subtype'] in ['fraction','whole','negint'] else str(round(x,6))
            except:
                D.loc[p,'acc']  = 0
                D.at[p,'ans']   = str(answer)
            if record_rules is not None:
                used_names = list(set([r.name for s, r in M.trace if r is not None and r.name in record_names]))
                D.loc[p, used_names] = 1
        if verbose: print()
        self.doOnEnd(M)
        return(D)
    def doOnStart(self, M):
        pass
    def doOnIter(self, M, i):
        pass
    def doOnEnd(self, M):
        pass
    def tabulate(self, cols=None):
        if cols is None: cols = ['notation', 'operation']
        return(pd.crosstab(index=self.P[cols], columns='count'))

class Course(ProblemSet):
    """ A Course is a ProblemSet intended to be used for training. It will run models in learning mode, and will make some rules unavailable to models until they encounter a problem for which the rules would normally be taught. So, e.g., decimal multiplication procedures won't be available when decimal addition is encountered for the first time, but will become available once decimal multiplication has been encountered. """
    def __init__(self, train_set, filter=None, N=None, mode="COMPLETE_COURSE"):
        super(Course, self).__init__(train_set, filter, N)
        self.mode           = mode          # see doOnStart for explanation & possible values of self.mode
        self.T, self.rules  = self.makeTeachingPlan(self.P)
    def __str__(self):
        return(str([str([rule.name for rule in U]) for U in self.T]))
    def makeTeachingPlan(self, P):
        """ Create a list containing, for each problem in P, a list of rules to be taught prior to solving the problem """
        CR = {
            ('fraction',):              [R_ONOD_OG],                # fraction indep comp strategy (whole number bias)
            ('fraction', 'add', 'ED'):  [R_KDON_AS, R_KDON_OG],     # fraction add/sub ED rules
            ('fraction', 'sub', 'ED'):  [R_KDON_AS, R_KDON_OG],     # 
            ('fraction', 'add', 'UD'):  [R_CDON_AS, R_CDON_OG],     # fraction add/sub UD rules
            ('fraction', 'sub', 'UD'):  [R_CDON_AS, R_CDON_OG],     # 
            ('fraction', 'mul'):        [R_ONOD_M, R_CROP_M],       # fraction multiplication rules
            ('fraction', 'div'):        [R_ICDM_D, R_ICDM_OG],      # fraction division rules
            ('decimal', 'mul'):         [R_ARAD_M, R_ARAD_OG] }     # dec mul rules are inaccessible until dec mul is encountered
        # CR rules is the set of rules that appear in CR
        CR_rules            = set([rule for context, rules in CR.items() for rule in rules])
        # RC contains rule name, contexts pairs, where contexts is a list of contexts in which the named rule appears in CR
        RC                  = {rule.name: [context for context, rules in CR.items() if rule.name in [r.name for r in rules]] for rule in CR_rules}
        # for each problem in P, create a list of rules that should be taught immediately prior to the problem
        T = [[] for i in range(P.shape[0])]
        for rule in CR_rules:
            contexts = RC[rule.name]
            for i in range(P.shape[0]):
                match       = any([set(context).issubset(set(P.iloc[i])) for context in contexts])
                if match:
                    T[i].append(rule)
                    break
        return(T, CR_rules)
    def trainModel(self, M, rulesToAdd=None, force_rules=None):
        D = self.runModel(M, learn=True, verbose=False, force_rules=force_rules)
    def doOnStart(self, M):
        if self.mode=="COMPLETE_COURSE":
            # Rules in both model and teaching plan are dropped by doOnStart
            # Rules that were dropped are restored either by addOnIter or addOnEnd
            # Thus, final model should have same rule set as original model
            self.dropOnStart = [rule for rule in self.rules if rule.name in [r.name for r in M.rules]]
            self.addOnIter   = [r.name for r in self.dropOnStart]
            self.addOnEnd    = self.dropOnStart
        elif self.mode=="FIRST_COURSE":
            # Rules in teaching plan are dropped by doOnStart (no effect if rule not in model)
            # Rules in teaching plan are added (if applicable) by addOnIter
            # Rules not added by addOnIter are NOT restored by addOnEnd (should be done by LATER_COURSE)
            # This avoids adding rules that shouldn't be added until later courses
            # NOTE: addOnIter can add rules that were not in the original model, so be careful!
            self.dropOnStart = self.rules
            self.addOnIter   = [r.name for r in self.rules]
            self.addOnEnd    = []
        elif self.mode=="LATER_COURSE":
            # Like FIRST_COURSE, except:
            # No rules are dropped by doOnStart (should have been done by FIRST_COURSE)
            # This avoids dropping rules that were added by earlier courses
            # NOTE: Make sure rules dropped on start of FIRST_COURSE are restored by addOnIter in FIRST_COURSE or some LATER_COURSE
            self.dropOnStart = []
            self.addOnIter   = [r.name for r in self.rules]
            self.addOnEnd    = []
        for rule in self.dropOnStart:
            M.dropRule(rule)
    def doOnIter(self, M, i):
        """ Add rules in teaching plan for current problem if they're in self.addOnIter """
        for rule in self.T[i]:
            if rule.name in self.addOnIter:
                M.addRule(rule)
    def doOnEnd(self, M):
        for rule in self.addOnEnd:
            M.addRule(rule)

class Test(ProblemSet):
    """ A Test is a ProblemSet intended to be used for testing. It will run models with learning off, add diagnostic info to output of model runs, and aggregates results from assessing the same model multiple times. """
    def __init__(self, test_set, filter=None, record_rules=None):
        super(Test, self).__init__(test_set, filter=filter)
        self.record_rules   = record_rules
    def testModel(self, M, force_rules=None):
        D = self.runModel(M, learn=False, verbose=False, force_rules=force_rules, record_rules=self.record_rules)
        return(D)

class Cohort():
    """ A Cohort generates a group of models. """
    def __init__(self, rules=all_rules, params_list=None, params_sets=None, N=1):
        if params_list is not None:
            # params_list is a list of dicts representing param settings for UMA instances
            pass
        elif params_sets is not None:
            # generate N instances of UMA for each combination of param vals listed in params_sets
            params_list = Cohort.makeParamsListFromSets(params_sets, N)
        self.subjids    = [i for i in range(len(params_list))]
        self.params     = params_list
        self.rules      = rules
        self.students   = [UMA(rules=self.rules, params=p) for p in self.params]
        for i in range(len(self.subjids)):
            if 'ice' in self.params[i].keys():
                self.students[i].reinforceRule(self.params[i]['ice'], R_acc_once, features=['+', 'whole'])
    @classmethod
    def makeParamsListFromSets(cls, params_sets={}, N=1):
        """ Generate params_list by equally representing all combinations of values in params_sets """
        # params_sets is a dict of form {param: vals} where vals is a list of values for param
        # N instances will be generated for each possible combination of param vals
        params_list = list(dict(zip(params_sets.keys(), values)) for values in product(*params_sets.values()))
        default_vals = {
            'g':            .06,
            'd':            0.5,
            'c':            4.0,
            'rt_mu':        4.5,
            'rt_sd':        1.0,
            'ice':          0 }
        for i in range(len(params_list)):
            params_list[i] = {**default_vals, **params_list[i]}
        params_list = [p for n in range(N) for p in params_list]
        return(params_list)
    @classmethod
    def saveModels(self, fn="models"):
        f = os.path.join(os.path.abspath(os.getcwd()), "Output", fn + ".xlsx")
        with pd.ExcelWriter(f, engine="xlsxwriter") as writer:
            workbook   = writer.book
            S = pd.DataFrame([[s, json.dumps(p)] for (s,p) in zip(self.subjids, self.params)], columns=['subjid', 'params'])
            S.to_excel(writer, sheet_name='subj')
            P = None
            A = None
            for subjid, params, student in zip(self.subjids, self.params, self.students):
                Q = student.proc_mem.reset_index()
                Q.insert(0, 'subjid', subjid)
                B = student.ans_mem.reset_index()
                B.insert(0, 'subjid', subjid)
                if P is None:
                    P = Q
                    A = B
                else:
                    P = pd.concat([P, Q], ignore_index=True)
                    A = pd.concat([A, B], ignore_index=True)
            P.to_excel(writer, sheet_name='proc')
            A.to_excel(writer, sheet_name='ans')
    @classmethod
    def loadModels(cls, fn, rules=all_rules):
        f = os.path.join(os.path.abspath(os.getcwd()), "Output", fn + ".xlsx")
        S = pd.read_excel(f, sheet_name='subj', index_col=0)
        P = pd.read_excel(f, sheet_name='proc', index_col=0)
        A = pd.read_excel(f, sheet_name='ans', index_col=0)
        rules_list  = [rule for rule in rules if rule.name in P.columns]
        params_list = [json.loads(p) for p in S['params']]
        C = Cohort(rules=rules_list, params_list=params_list)
        for subjid, student in zip(C.subjids, C.students):
            Q = P[P['subjid']==subjid]
            Q = Q.drop(['subjid'], axis=1)
            Q = Q.set_index('index')
            student.proc_mem = Q
            B = A[A['subjid']==subjid]
            B = B.drop(['subjid'], axis=1)
            B = B.set_index('index')
            student.ans_mem = B
        return(C)
    def saveResults(self, fn="results", add_date=True):
        out = None
        if self.results is not None:
            for subjid, params, result in zip(self.subjids, self.params, self.results):
                R = result.copy()
                orig_cols = R.columns
                R['subjid'] = subjid
                for param, value in params.items():
                    R[param] = value
                R = R[['subjid']+list(params.keys())+list(orig_cols)]
                if out is None:
                    out = R
                else:
                    out = pd.concat([out, R], ignore_index=True)
        f = os.path.join(os.path.abspath(os.getcwd()), "Output", fn)
        if add_date:
            f = f + " " + datetime.now().strftime("%y-%m-%d %H.%M.%S")
        f = f + ".xlsx"
        out.to_excel(f)

class Simulation():
    def __init__(self,
        # file name stem
        name,
        # cohort params
        params_sets     = None,
        rules           = None,
        N               = 1,
        # training and test params
        curriculum      = "full"):
    
        ## Save sim name and create cohort
        self.name   = name
        if rules is not None:
            base_rules  = rules
        elif curriculum=="full":
            base_rules  = all_rules
        elif curriculum in ["rational", "fractions", "decimals"]:
            base_rules  = RA_rules
        self.cohort = Cohort(rules=base_rules, params_sets=params_sets, N=N)
        
        ## Define self.courses
        if curriculum=="full":
            text        = "go_math"
            start_grade = 1
            end_grade   = 6
        elif curriculum in ["rational", "fractions", "decimals"]:
            text        = "go_math_" + curriculum
            start_grade = 6
            end_grade   = 6
        P       = pd.read_csv('Problem Sets Training/'+text+'.csv')
        volumes = [P[P['grade']==(i+1)]['prob'].tolist() for i in range(6)]
        self.courses    = []
        for grade in range(1, 7):
            if (grade < start_grade) | (grade > end_grade):
                self.courses.append(None)
            elif grade == start_grade:
                self.courses.append(Course(volumes[grade-1], mode="FIRST_COURSE"))
            else:
                self.courses.append(Course(volumes[grade-1], mode="LATER_COURSE"))
                
        ## Define self.tests
        wn_strats   = [R_retrieve, R_add_count, R_mul_rep_add, R_acc_once, R_acc_skip, R_acc_extra]
        da_strats   = [R_ADBD_AS, R_ADBD_OG, R_ARAD_M, R_ARAD_OG, R_VM_shift2, R_VM_shift2_no_zeros]
        fa_strats   = [R_KDON_AS, R_KDON_OG, R_CDON_AS, R_CDON_OG, R_ONOD_M, R_ONOD_OG, R_CROP_M, R_ICDM_D, R_ICDM_OG]
        TSA = Test("sd_add", record_rules=wn_strats)
        TSM = Test("sd_mul", record_rules=wn_strats)
        TDA = Test("bss2021", filter=["add"], record_rules=da_strats)
        TDM = Test("bss2021", filter=["mul"], record_rules=da_strats)
        TFA = Test("sp2013", record_rules=fa_strats)
        self.tests = [
            [TSA],                      # Grade 1
            [TSA],                      # Grade 2
            [TSA, TSM],                 # Grade 3
            [TSA, TSM],                 # Grade 4
            [TSA, TSM],                 # Grade 5
            [TSA, TSM, TDA, TDM, TFA]]  # Grade 6
        if curriculum in ["rational", "fractions", "decimals"]:
            for i in range(5):
                self.tests[i] = None
            if curriculum=="rational":
                self.tests[5] = [TDA, TDM, TFA]
            elif curriculum=="decimals":
                self.tests[5] = [TDA, TDM]
            elif curriculum=="fractions":
                self.tests[5] = [TFA]
    def run(self, start_subjid=0, end_subjid=None, save_final_only=False):
        """
        Run simulation for all students.

        Args:
            start_subjid: First student index to process (for parallelization)
            end_subjid: Last student index to process (for parallelization)
            save_final_only: If True, only save grade 6 model and test files (reduces ~19k to ~6k files)
        """
        # create directory to save results
        fn_stem = os.path.join(os.path.abspath(os.getcwd()), "Output", self.name) + os.sep
        if not os.path.exists(fn_stem):
            os.makedirs(fn_stem)
        time1 = datetime.now()
        # record start time
        print("Starting sim " + self.name)
        print(time1.strftime("%y-%m-%d %H.%M.%S"))
        # run simulation
        for subjid in range(start_subjid, len(self.cohort.subjids)):
            if end_subjid is None or subjid<=end_subjid:
                print("Starting subject " + str(subjid+1) + " of " + str(len(self.cohort.subjids)))
                for grade in range(1, 7):
                    course  = self.courses[grade-1]
                    tests   = self.tests[grade-1]
                    sfn_stem = fn_stem + "sim subjid_" + str(subjid) + " grade_" + str(grade)
                    # train model
                    if course is not None:
                        course.trainModel(self.cohort.students[subjid])
                        # Only save if final grade OR not in save_final_only mode
                        if grade == 6 or not save_final_only:
                            with pd.ExcelWriter(sfn_stem + " model.xlsx", engine="xlsxwriter") as writer:
                                S = pd.DataFrame([[subjid, json.dumps(self.cohort.params[subjid])]], columns=['subjid', 'params'])
                                S.to_excel(writer, sheet_name='subj')
                                self.cohort.students[subjid].proc_mem.to_excel(writer, sheet_name='proc')
                                self.cohort.students[subjid].ans_mem.to_excel(writer, sheet_name='ans')
                    # test model
                    if tests is not None:
                        # Only save if final grade OR not in save_final_only mode
                        if grade == 6 or not save_final_only:
                            for j in range(len(tests)):
                                R = tests[j].testModel(self.cohort.students[subjid])
                                orig_cols = R.columns
                                R['subjid'] = subjid
                                for param, value in self.cohort.params[subjid].items():
                                    R[param] = value
                                R = R[['subjid']+list(self.cohort.params[subjid].keys())+list(orig_cols)]
                                R.to_excel(sfn_stem + " test_" + str(j+1) + ".xlsx")
                    self.cohort.students[subjid].trace = [] # maybe saves memory?
        # record end time and run time
        time2 = datetime.now()
        print(time2.strftime("%y-%m-%d %H.%M.%S"))
        print(time2 - time1)
    @classmethod
    def mergeResults(cls, name):
        """ Combine results of named simulation into a small number of Excel files """
        # get filenames
        path = os.path.join(os.path.abspath(os.getcwd()), "Output", name) + os.sep
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        P = pd.DataFrame(files, columns=['file'])
        for c in ['subjid', 'grade', 'type', 'number']:
            P[c] = ""
        for i in P.index:
            f = P.loc[i,'file']
            P.loc[i,'subjid']   = re.compile(r"subjid_[0-9]+").search(f).group().replace("subjid_", "")
            P.loc[i,'grade']    = re.compile(r"grade_[0-9]+").search(f).group().replace("grade_", "")
            if "model" in f:
                P.loc[i,'type'] = "model"
                P.loc[i,'test'] = ""
            elif "test" in f:
                P.loc[i,'type'] = "test"
                P.loc[i,'test'] = re.compile(r"test_[0-9]+").search(f).group().replace("test_", "")
        # merge results
        X = None
        for test in set(P['test']):
            if test!="":
                Q = P[P['test']==test]
                for grade in set(Q['grade']):
                    R = Q[Q['grade']==grade]
                    for subjid in set(R['subjid']):
                        file = list(R[R['subjid']==subjid]['file'])[0]
                        if X is None:
                            X = pd.read_excel(file)
                            X.insert(0, 'grade', grade)
                            X.insert(1, 'test', test)
                        else:
                            Y = pd.read_excel(file)
                            Y.insert(0, 'grade', grade)
                            Y.insert(1, 'test', test)
                            X = pd.concat([X, Y], ignore_index=True)
        X = X.drop(columns=[c for c in X.columns if "Unnamed" in c])
        X['notation'] = pd.Categorical(list(X['notation']), categories=["whole", "fraction", "decimal"])
        X = X.sort_values(by=['notation', 'grade', 'test', 'subjid'])
        X.to_csv(os.path.join(os.path.abspath(os.getcwd()), "Output", "sim " + name + ".csv"))

# # Uncomment these lines to run the full simulations
# ps      = {
#    'g':     [.01, .02, .03, .04, .05, .06, .07, .08, .09, .10],
#    'd':     [.1, .3, .5, .7, .9],
#    'c':     [5],
#    'rt_mu': [3, 4, 5, 6],
#    'rt_sd': [1],
#    'ice':   [0, 25, 50, 75, 100]}
# S = Simulation("ALL29_BCD", params_sets=ps, rules=all_rules, N=1, curriculum="full")
# S.run()
# Simulation.mergeResults("ALL29_BCD")

# # Uncomment these lines to run a simulation with one simulated student
# ps      = {
   # 'g':     [.05],
   # 'd':     [.3],
   # 'c':     [5],
   # 'rt_mu': [5],
   # 'rt_sd': [1],
   # 'ice':   [50]}
# S = Simulation("XXX", params_sets=ps, rules=all_rules, N=1, curriculum="full")
# S.run()
# Simulation.mergeResults("XXX")
