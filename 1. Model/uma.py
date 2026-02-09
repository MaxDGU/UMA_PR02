# -*- coding: utf-8 -*-

# A Unified Model of Arithmetic (UMA) Version 0.600
# By David W. Braithwaite (baixiwei@gmail.com; braithwaite@psy.fsu.edu)

import numpy as np, pandas as pd
from scipy.special import softmax
import os, copy, sys, warnings, time, collections, re

uma_version = "0.600"

UMA_traces = []

def myEval(p):
    """ Calculate the value of an arithmetic expression in the form of a string """
    s = str(p)
    # replace mixed numbers with parenthesized sums
    mixed = re.compile(r"\d+\s\d+/\d+")
    while mixed.search(s):
        mixed_num = mixed.search(s).group()
        mixed_parts = mixed_num.split(" ")
        mixed_sum = "(" + mixed_parts[0] + "+" + mixed_parts[1] + ")"
        s = s.replace(mixed_num, mixed_sum)
    if ":" in s:
        s = s.split(":")
        x = eval(s[0])/eval(s[1])
    else:
        x = eval(s)
    return(int(x) if int(x)==x else x)


### Memory Structures: States, Contexts, and Chunks

# State
#   Represents UMA's state at one step of a problem-solving episode
#   Goal stack (gs) determines which rule(s) can be used right now
#   Context stack (cs) represents features of problems UMA is working on;
#   top Context in the stack determines current activations of rules and answers
#   Workspace (ws) represents Chunks (e.g., Problems) held in STM or written down
#   Retrieved answer (ra) is a memory buffer to contain output of memory retrieval

class State:
    def __init__(self, p=None):
        self.gs     = []
        self.cs     = []
        self.ws     = []
        self.ra     = None
        if p is None:
            self.cs.append(Context())
        else:
            self.addProb(p)
    def __str__(self):
        s = ""
        if len(self.gs)>0:
            s += "GOALS: " + str([g.features['name'] for g in self.gs]) + "\n"
            s += "CURRENT GOAL: " + str(self.gs[-1]) + "\n"
        if len(self.cs)>0:
            s += "CURRENT CONTEXT: " + str(self.cs[-1].getFeatures()) + "\n"
        return(s)
    def addProb(self, p):
        problem     = p if isinstance(p, Problem) else Problem(p)
        goal        = Goal('solve_problem', {'problem': problem})
        self.pushGoal(goal)
        self.ws.append(problem)
    def addChunk(self, c):
        self.ws.append(c)
    def pushGoal(self, goal):
        if goal.features['name']=='solve_problem':
            self.cs.append(Context(goal.features['problem']))
        self.gs.append(goal)
    def popGoal(self):
        if (len(self.gs)>0):
            if self.gs[-1].features['name']=='solve_problem':
                self.cs.pop()
            return(self.gs.pop())
        else:
            return(None)
    def getGoal(self):
        if (len(self.gs)>0):
            return(self.gs[-1])
        else:
            return(None)
    def addRetrievedAnswer(self, answer):
        self.ra = answer
    def getRetrievedAnswer(self):
        return(self.ra)
    def clearRetrievedAnswer(self, answer):
        self.ra = None
    def getContext(self):
        if (len(self.cs)>0):
            return(self.cs[-1])
        else:
            return(None)

# Context
#   Vector representation of a Problem's features using fixed dimensions as below
#   Context converts problems into feature vectors using these dimensions
#   gen_features are problem features, a subset of which are used as UMA's proc_mem indices
#   mem_probs are literal problems, which correspond to the set of problems UMA can retrieve from memory and also serve as UMA's ans_mem indices

operations      = ['+', '-', '*', ':']
number_types    = ['whole', 'fraction', 'decimal']
num_subtypes    = ['SD', 'MD', 'FR', 'MI', 'WF', 'ED', 'UD', 'WD', 'DD', 'EDD', 'UDD']
gen_features    = operations + number_types + num_subtypes
mem_probs       = [str(i)+op+str(j)  for op in ['+', '*'] for i in range(1,11) for j in range(1,11)]
mem_answers     = [str(i) for i in range(1,101)]

class Context:
    def __init__(self, x=None):
        self.prob = ""
        self.data = pd.Series(0, index=gen_features + mem_probs)
        if type(x) is Context:
            self.prob = x.prob
            self.data = x.data.copy()
        elif x is not None:
            v   = []
            if type(x) in [Problem, vProblem]:
                pf = x.features
            else:
                pf = Problem(x).features
            o1  = Number() if pf['operand1'] is None else pf['operand1']
            o2  = Number() if pf['operand2'] is None else pf['operand2']
            o1f = o1.features
            o2f = o2.features
            self.prob = str(o1)+pf['operator']+str(o2)
            # arithmetic operation
            v = [pf['operator']]
            # number types and number subtypes
            if 'fraction' in [o1f['subtype'], o2f['subtype']] or 'mixed' in [o1f['subtype'], o2f['subtype']]:
                v.append('fraction')
                # subtype: FR, MI, or WF
                if 'whole' in [o1f['subtype'], o2f['subtype']]:
                    v.append('WF')
                elif 'mixed' in [o1f['subtype'], o2f['subtype']]:
                    v.append('MI')
                else:
                    v.append('FR')
                # subtype: ED or UD
                if o1f['subtype']=='fraction':
                    den1 = int(o1f['den'])
                elif o1f['subtype']=='mixed':
                    den1 = int(o1f['fraction'].features['den'])
                elif o1f['subtype']=='whole':
                    den1 = 1
                if o2f['subtype']=='fraction':
                    den2 = int(o2f['den'])
                elif o2f['subtype']=='mixed':
                    den2 = int(o2f['fraction'].features['den'])
                elif o2f['subtype']=='whole':
                    den2 = 1
                if den1==den2:
                    v.append('ED')
                else:
                    v.append('UD')
            elif 'decimal' in [o1f['subtype'], o2f['subtype']]:
                v.append('decimal')
                # subtype: DD or WD
                if 'whole' in [o1f['subtype'], o2f['subtype']]:
                    v.append('WD')
                elif o1f['subtype']=='decimal' and o2f['subtype']=='decimal':
                    v.append('DD')
                # subtype: EDD or UDD
                if o1f['subtype']=='decimal':
                    dd1 = int(o1f['num_dec_dig'])
                elif o1f['subtype']=='whole':
                    dd1 = 0
                if o2f['subtype']=='decimal':
                    dd2 = int(o2f['num_dec_dig'])
                elif o2f['subtype']=='whole':
                    dd2 = 0
                if dd1==dd2:
                    v.append('EDD')
                else:
                    v.append('UDD')
            elif (o1f['subtype']=='whole' and o2f['subtype'] in ['whole', None]) or (o2f['subtype']=='whole' and o1f['subtype'] in ['whole', None]):
                v.append('whole')
                # subtype: SD (both operands<=10) or MD (otherwise)
                if (o1f['subtype']=='whole' and int(o1)>10) or (o2f['subtype']=='whole' and int(o2)>10):
                    v.append('MD')
                else:
                    v.append('SD')
            # specific problem
            v = v + [self.prob]
            self.data[[f for f in self.data.index if f in v]] = 1
    def __str__(self):
        return(str(self.getFeatures()))
    def __eq__(self, c):
        return(self.data.equals(c.data))
    def __ne__(self, c):
        return(not self.data.equals(c.data))
    def getFeatures(self):
        return(list(self.data[self.data==1].index))

# Calculate Euclidean distance between pairs of mem_probs

def getProbDist(p, q):
    """ Calculate Euclidean distance between problems p and q
    Distance is number of differences squared with respect to operands and operator """
    Poper = [x for x in operations if x in p][0]
    Pop1, Pop2 = p.split(Poper)
    Qoper = [x for x in operations if x in q][0]
    Qop1, Qop2 = q.split(Qoper)
    diff = 0 + (Poper!=Qoper)
    if ((Pop1==Qop1) and (Pop2==Qop2)) or ((Pop1==Qop2) and (Pop2==Qop1)):
        # both operands same in any order => no difference
        pass
    elif (Pop1==Qop2) or (Pop2==Qop1):
        # one operand same in any order => one difference
        diff += 1
    else:
        # neither operand same => two differences
        diff += 2
    dist = diff ** 2
    return(dist)

mem_prob_dists = pd.DataFrame(
    [[getProbDist(p, q) for q in mem_probs] for p in mem_probs],
    index=mem_probs, columns=mem_probs)


# Chunk
#   A representational structure that UMA rules can create, access, and modify
#   represents a categorical feature vector, i.e. a set of slot: value pairs
#   is stored as a Python dict, with dict keys corresponding to Chunk slots
# Chunk subtypes
#   represent particular types of mathematical object like Numbers, Problems, etc.
#   guaranteed to have certain features (i.e., slots/dict keys) but vals can be None

class Chunk:
    def __init__(self, features={}):
        self.features = features
    def __str__(self):
        return(str([(key, str(val)) for (key, val) in self.features.items()]))
    def __eq__(self, c):
        if type(self)!=type(c):
            return(False)
        else:
            return(self.features==c.features)
    def addFeatures(self, features):
        try:
            self.features = {**self.features, **features}
        except:
            print("Error in Chunk.addSlotsVals: %s %s" % str(self.features), features)
    def delFeatures(self, features, leave_slot=True):
        for slot in features.keys():
            if slot in self.features:
                if leave_slot:
                    self.features[slot] = None
                else:
                    del self.features[slot]

class Goal(Chunk):
    def __init__(self, name=None, features={}):
        super(Goal, self).__init__({**{'type': 'goal', 'name': name}, **features})

def toDigArr(n):
    """ Interpret n as an array of digits """
    if type(n)==list and all(type(d) in [int, np.int32, np.int64] for d in n):
        # n is already an array of digits
        return(n)
    elif type(n) in [int, np.int32, np.int64]:
        # n is an integer
        return [int(d) for d in str(n)]
    elif type(n)==list and all((type(k)==str and k.isdigit() and len(k)==1) for k in n):
        # n is an array of strings representing digits
        return([int(d) for d in n])
    elif type(n)==str and n.isdigit():
        # n is a single string of digits
        return([int(d) for d in n])
    elif isinstance(n, Chunk) and 'digits' in n.features.keys():
        # n is a chunk that has a feature 'digits'
        return([] if n.features['digits'] is None else n.features['digits'])
    else:
        print("Unrecognized input to toDigArr: %s" % str(n))
        return(n)

class Number(Chunk):
    def __init__(self, number=None, features={}):
        super(Number, self).__init__({**{'type': 'number', 'subtype': None}, **features})
        
        if isinstance(number, Number):
            self.features = copy.deepcopy(number.features)
        
        # try to set n as a str representing number
        n = None
        if type(number)==str:
            n = number
        elif type(number) in [int, np.int32, np.int64]:
            n = str(number)
        elif type(number)==list and all(type(k) in [int, np.int32, np.int64] for k in number):
            n = "".join([str(d) for d in number])
        elif type(number)==list and all(type(k)==str and k.isdigit() for k in number):
            n = "".join([d for d in number])
        
        # if n is a str, try to identify subtype and set subtype features
        if type(n)==str:
            f = n.split("/")
            d = n.split(".")
            if n.isdigit():
                self.features['subtype']    = 'whole'
                self.features['digits']     = [int(d) for d in n]
                self.features['multidig']   = len(self.features['digits'])>1
                self.features['num_dec_dig'] = None
            elif len(n)>0 and n[0]=="-" and n[1:].isdigit():
                self.features['subtype']    = 'negint'
                self.features['digits']     = [int(d) for d in n[1:]]
                self.features['multidig']   = len(self.features['digits'])>1
                self.features['num_dec_dig'] = None                
            # elif len(f)==2 and all(i.isdigit() for i in f):
            elif len(f)==2 and all(i.replace('.','').replace('-','').isdigit() for i in f):
                # this allows negatives and decimals in num and/or den
                self.features['subtype']    = 'fraction'
                self.features['num']        = Number(f[0])
                self.features['den']        = Number(f[1])
            elif re.compile(r"^\d+\s\d+/\d+$").match(n):
                self.features['subtype']    = 'mixed'
                parts = n.split(" ")
                self.features['whole']      = Number(parts[0])
                self.features['fraction']   = Number(parts[1])
            elif len(d)==2 and (d[0]=="" or d[0].isdigit()) and (d[1]=="" or d[1].isdigit()):
                self.features['subtype']    = 'decimal'
                self.features['digits']     = [int(i) for i in d[0]+d[1]]
                self.features['multidig']   = len(self.features['digits'])>1
                self.features['whole_part'] = None if d[0]=="" else Number(d[0])
                self.features['dec_part']   = None if d[1]=="" else Number(d[1])
                self.features['dec_pos']    = len(d[0])
                self.features['num_dec_dig'] = Number(0 if self.features['dec_part'] is None else len(self.features['dec_part'].features['digits']))
            # else:
                # print("Unrecognized input type to Number constructor: %s" % n)
            
        # if subtype given and expected features not set (e.g., in a pattern), set them to None
        if self.features['subtype'] in ['whole', 'negint']:
            for k in ['digits', 'multidig', 'num_dec_dig']: #, 'num_dig'
                if k not in self.features.keys():
                    self.features[k] = None
        elif self.features['subtype']=='decimal':
            for k in ['digits', 'multidig', 'whole_part', 'dec_part', 'dec_pos', 'num_dec_dig']:
                if k not in self.features.keys():
                    self.features[k] = None
        elif self.features['subtype']=='fraction':
            for k in ['num', 'den']:
                if k not in self.features.keys():
                    self.features[k] = None
        elif self.features['subtype']=='mixed':
            for k in ['whole', 'fraction']:
                if k not in self.features.keys():
                    self.features[k] = None
                    
        # set focus digit to None
        self.focus = None
    def __str__(self):
        if self.features['subtype']=='whole':
            if self.features['digits'] is None:
                return("WHOLE")
            else:
                return("".join([str(d) for d in self.features['digits']]))
        elif self.features['subtype']=='negint':
            if self.features['digits'] is None:
                return("NEGINT")
            else:
                return("-"+"".join([str(d) for d in self.features['digits']]))
        elif self.features['subtype']=='decimal':
            if self.features['whole_part'] is None and self.features['dec_part'] is None:
                return("DECIMAL")
            else:
                s = "."
                wp = self.features['whole_part']
                dp = self.features['dec_part']
                if wp is not None: s = str(wp) + s
                if dp is not None: s = s + str(dp)
                return(s)
        elif self.features['subtype']=='fraction':
            if self.features['num'] is None and self.features['den'] is None:
                return("FRACTION")
            else:
                return(str(self.features['num'])+"/"+str(self.features['den']))
        elif self.features['subtype']=='mixed':
            if self.features['whole'] is None and self.features['fraction'] is None:
                return("MIXED")
            else:
                return(str(self.features['whole'])+" "+str(self.features['fraction']))                
        else:
            return("NUMBER")
    def __int__(self):
        try:
            return(int(str(self)))
        except Exception as e:
            # print("Cannot convert Number to int: %s" % str(self))
            raise TypeError(f"Cannot convert Number to int: {str(self)}") from e
    def __float__(self):
        try:
            if self.features['subtype']=='fraction':
                return(float(float(self.features['num'])/float(self.features['den'])))
            elif self.features['subtype']=='mixed':
                return(float(self.features['whole']) + float(self.features['fraction']))
            else:
                return(float(str(self)))
        except Exception as e:
            # print("Cannot convert Number to float: %s" % str(self)) # comment out for long sims
            raise TypeError(f"Cannot convert Number to float: {str(self)}") from e
    def increment(self, by=1):
        try:
            if self.features['subtype']=="whole":
                old_dig = len(self.features['digits'])
                new_val = str(int(self) + int(by))
                self.features['digits']     = [int(d) for d in new_val]
                self.features['multidig']   = len(self.features['digits'])>1
                if self.focus is not None:
                    self.focus = self.focus + len(self.features['digits']) - old_dig
            else:
                print("Invalid Number for increment: " + str(self))
        except:
            print("Invalid Number for increment: " + str(self))
    def decrement(self, by=1):
        try:
            if self.features['subtype']=="whole":
                old_dig = len(self.features['digits'])
                new_val = str(int(self) - by)
                self.features['digits']     = [int(d) for d in new_val]
                self.features['multidig']   = len(self.features['digits'])>1
                if self.focus is not None:
                    self.focus = max(0, self.focus + len(self.features['digits']) - old_dig)
            else:
                print("Invalid Number for decrement: " + str(self))
        except:
            print("Invalid Number for decrement: " + str(self))
    def addDigits(self, digits, how="prepend"):
        if self.features['subtype'] is None or (self.features['subtype'] in ['whole', 'negint'] and self.features['digits'] is None):
            self.__init__(digits)
        elif self.features['subtype'] in ['whole', 'negint']:
            curr_digits = toDigArr(self)
            new_digits  = toDigArr(digits)
            if how=="prepend":
                self.features['digits'] = new_digits + curr_digits
                if self.focus is not None:
                    self.attendDig(self.focus + 1)
            elif how=="append":
                self.features['digits'] = curr_digits + new_digits
            self.features['multidig']   = len(self.features['digits'])>1
        elif self.features['subtype']=='decimal':
            curr_digits = toDigArr(self)
            new_digits  = toDigArr(digits)
            if how=="prepend":
                self.features['digits']     = new_digits + curr_digits
                self.features['dec_pos']    += len(new_digits)
                self.features['whole_part'] = Number(self.features['digits'][:self.features['dec_pos']])
                if self.focus is not None:
                    self.attendDig(self.focus + 1)
            elif how=="append":
                self.features['digits']     = curr_digits + new_digits
            self.features['multidig']   = len(self.features['digits'])>1
            self.features['dec_part']   = Number(self.features['digits'][self.features['dec_pos']:])
            self.features['num_dec_dig'] = Number(0 if self.features['dec_part'] is None else len(self.features['dec_part'].features['digits']))
        else:
            print("Cannot add digits to this Number subtype: %s %s %s" % (str(self.features['subtype']), str(self), str(digits)))
    def prependDigits(self, digits):
        self.addDigits(digits, "prepend")
    def appendDigits(self, digits):
        self.addDigits(digits, "append")
    def attendDig(self, pos):
        c = self.features['digits']
        # set self.focus
        if c is None or pos=="none":
            self.focus = None
        elif pos=="first":
            self.focus = 0
        elif pos=="last":
            self.focus = len(c)-1
        elif type(pos) in [int, np.int32]:
            self.focus = pos
        # set self.features['current']
        if self.focus is None or self.focus<0 or self.focus>=len(c):
            self.features['current'] = None
        else:
            self.features['current'] = Number(c[self.focus])
    def shiftDig(self, dir):
        if self.focus is not None:
            pos = self.focus
            if dir=="prev":
                pos = pos - 1
            elif dir=="next":
                pos = pos + 1
            self.attendDig(pos)

class Accumulator(Chunk):
    def __init__(self, features={}):
        super(Accumulator, self).__init__({
            **{
                'total':    None,   # running total (amount accumulated so far)
                'counter':  None,   # number of times you have accumulated
                'target':   None,   # number of times you are supposed to accumulate
                'action':   None,   # how to accumulate (e.g., 'count', 'add')
                'amount':   None    # how much to accumulate each time
            }, **features})
        for v in ['counter', 'target', 'total', 'amount']:
            if self.features[v] is not None and type(self.features[v]) is not Variable and type(self.features[v]) is not Number:
                self.features[v] = Number(self.features[v])
    def __str__(self):
        return('Accumulator ' + str(self.features['total']) + ' ' 
            + str(self.features['counter']) + '/' 
            + str(self.features['target']) + ' ' 
            + str(self.features['action']) + ' ' 
            + str(self.features['amount']))

class Problem(Chunk):
    def __init__(self, problem=None, features={}):
        super(Problem, self).__init__({**{
            'type':         'problem',
            'operator':     None,
            'operand1':     None,
            'operand2':     None,
            'answer':       None}, **features})
        if type(problem) in [str, np.str_]:
            oper, op1, op2 = Problem.decomposeProbTxt(problem)
            self.features['operator'] = oper
            self.features['operand1'] = op1
            self.features['operand2'] = op2
    def __str__(self):
        v = self.features
        s = str(v['operand1']) + " " + str(v['operator']) + " " + str(v['operand2']) + " = "
        if self.features['answer'] is None:
            s += "?"
        else:
            s += str(v['answer'])
        return(s)
    def toStrProbOnly(self):
        return(str(self.features['operand1'])+self.features['operator']+str(self.features['operand2']))
    def decomposeProbTxt(txt):
        arithOperators = ["+","*","-",":"]
        try:
            operator = [x for x in arithOperators if x in txt][0]
            operands = txt.split(operator)
            operand1, operand2 = [Number(operand) for operand in operands]
            return(operator, operand1, operand2)
        except:
            print("Error in Problem.decomposeProbTxt: %s" % txt)
    def asFeatureVector(self, dims=None):
        operdim = self.features['operator']
        op1dim  = "op1_" + str(self.features['operand1'])
        op2dim  = "op2_" + str(self.features['operand2'])
        probdim = str(self.features['operand1'])+self.features['operator']+str(self.features['operand2'])
        nonzero = [operdim, op1dim, op2dim, probdim]
        dims    = nonzero if dims is None else dims
        vals    = [int(dim in nonzero) for dim in dims]
        return(pd.DataFrame([vals], columns=dims))
    def asFeatureList(self, dims=None):
        vector  = self.asFeatureVector(dims)
        return([dim for dim in vector.columns if vector.loc[0,dim]==1])
    
class List(Chunk):
    def __init__(self, contents=None, features={}):
        super(List, self).__init__(features={**{
            'type':         'list', 
            'contents':     contents,
            'length':       None,
            'empty':        None,
            'first':        None,
            'second':       None,
            'last':         None,
            'current':      None,
            'list_ans':     None        # store sum or product of list
            }, **features})
        self.focus = None
        self.updateFeatures()
    def __str__(self):
        if self.features['contents'] is None:
            return("")
        else:
            S = str([str(x) for x in self.features['contents']])
            if self.features['list_ans'] is not None:
                S = S + " = " + str(self.features['list_ans'])
            return(S)
    def prepend(self, x):
        self.features['contents'] = [x] if self.features['contents'] is None else [x] + self.features['contents']
        if self.focus is not None:
            self.focus = self.focus + 1
        self.updateFeatures()
    def append(self, x):
        self.features['contents'] = [x] if self.features['contents'] is None else self.features['contents'] + [x]
        self.updateFeatures()
    def attendPos(self, pos):
        c = self.features['contents']
        # set self.focus
        if c is None or pos=="none":
            self.focus = None
        elif pos=="first":
            self.focus = 0
        elif pos=="last":
            self.focus = len(c)-1
        elif type(pos) in [int, np.int32]:
            self.focus = pos
        # set self.features['current'] and ['attn_in']
        self.features['attn_in'] = not (self.focus is None or self.focus<0 or self.focus>=len(c))
        self.features['current'] = c[self.focus] if self.features['attn_in'] else None
    def shiftPos(self, dir):
        if self.focus is not None:
            pos = self.focus
            if dir=="prev":
                pos = pos - 1
            elif dir=="next":
                pos = pos + 1
            self.attendPos(pos)
    def updateFeatures(self):
        c = self.features['contents']
        if c is not None:
            self.features['length'] = len(c)
            self.features['empty']  = len(c)==0
            if len(c)>0:
                self.features['first']  = c[0]
                self.features['second'] = c[1] if len(c)>1 else None
                self.features['last']   = c[-1]
            self.features['attn_in'] = not (self.focus is None or self.focus<0 or self.focus>=len(c))
            self.features['current'] = c[self.focus] if self.features['attn_in'] else None

class arithMat(List):
    # arithMat: a representation of a List of Numbers written in vertical format
    # arithMat.cols contains a Python list of Lists of single digit Numbers representing aligned columns of digits
    # arithMat.focus_col contains the index of the currently attended column
    # arithMat.features['curr_col'] contains the currently attended column
    # empty spaces in columns are represented by None
    def __init__(self, contents=None, features={}):
        super(arithMat, self).__init__(contents=contents, features={**{'curr_col': None}, **features})
        if self.features['contents'] is None:
            self.align      = None
            self.width      = None
            self.offsets    = None
            self.rows       = None
            self.cols       = None
        else:
            self.align      = 'right'
            self.alignOperands('right')
        self.focus_col  = None
    def __str__(self):
        if self.features['contents'] is None:
            return("EMPTY ARITHMAT")
        else:
            s = "".join([" " if d is None else str(d) for d in self.rows[0]])
            for row, (lo, ro) in zip(self.rows[1:], self.offsets[1:]):
                s += "\n" + "".join([" " if d is None else str(d) for d in row])
            return(s)
    def alignOperands(self, align=None, append=False):
        """ Set self.width, self.offsets, and self.cols to reflect right, left, or decimal point alignment """
        c = self.features['contents']
        if c is not None:
            self.align = self.align if align is None else align
            if self.align=='right':
                self.width      = max([len(n.features['digits']) for n in c])
                self.offsets    = [(self.width - len(n.features['digits']), 0) for n in c]
            elif self.align=='left':
                self.width      = max([len(n.features['digits']) for n in c])
                self.offsets    = [(0, self.width - len(n.features['digits'])) for n in c]
            elif self.align=='decimal':
                if append:
                    # equalize numbers of decimal digits in operands
                    dec_ops = [n for n in c if n.features['subtype']=='decimal']
                    dec_dig = [len(n.features['dec_part'].features['digits']) for n in dec_ops]
                    max_dd  = max(dec_dig)
                    for i in range(len(c)):
                        if c[i].features['subtype']=='whole':
                            dd = 0
                        elif c[i].features['subtype']=='decimal':
                            dd = len(c[i].features['dec_part'].features['digits'])
                        if dd < max_dd:
                            if c[i].features['subtype']=='whole':
                                c[i] = Number(str(c[i]) + "." + "".join(['0' for i in range(max_dd)]))
                            elif c[i].features['subtype']=='decimal':
                                c[i] = Number(str(c[i]) + "".join(['0' for i in range(max_dd - dd)]))
                    self.updateFeatures()
                wd = []
                dd = []
                for n in c:
                    if n.features['subtype']=='whole':
                        wd.append(len(n.features['digits']))
                        dd.append(0)
                    elif n.features['subtype']=='decimal':
                        wd.append(0 if n.features['whole_part'] is None else len(n.features['whole_part'].features['digits']))
                        dd.append(len(n.features['dec_part'].features['digits']))
                self.width      = max(wd) + max(dd)
                lo = [max(wd)-w for w in wd]
                ro = [max(dd)-d for d in dd]
                self.offsets    = [(lo,ro) for (lo,ro) in zip(lo, ro)]
            self.rows   = []
            for n, (lo, ro) in zip(c, self.offsets):
                row = [None] * lo + [Number(d) for d in n.features['digits']] + [None] * ro
                self.rows.append(row)
            C = np.array(self.rows).T.tolist()
            self.cols = [List(col) for col in C]
    def getDecCols(self):
        """ Number of columns right of decimal point in each operand """
        c = self.features['contents']
        if c is not None:
            dc = []
            for n, (lo, ro) in zip(c, self.offsets):
                if n.features['subtype']=='whole':
                    dc.append(ro)
                else:
                    dc.append(ro + len(n.features['dec_part'].features['digits']))
            return(dc)
    def prepend(self, x):
        super().prepend(x)
        self.align = 'right' if self.align is None else self.align
        self.alignOperands()
    def append(self, x):
        super().append(x)
        self.align = 'right' if self.align is None else self.align
        self.alignOperands()
        if self.focus_col is not None:
            self.attendCol(self.focus_col)
    def attendCol(self, pos):
        # set self.focus_col
        if self.cols is None or pos=="none":
            self.focus_col = None
        elif pos=="first":
            self.focus_col = 0
        elif pos=="last":
            self.focus_col = len(self.cols)-1
        elif type(pos) in [int, np.int32]:
            self.focus_col = pos
        # set self.features['curr_col']
        if self.focus_col is None or self.focus_col<0 or self.focus_col>=len(self.cols):
            self.features['curr_col'] = None
        else:
            self.features['curr_col'] = self.cols[self.focus_col]
        # for each number in self, attend to digit nearest to current col
        for n, (lo, ro) in zip(self.features['contents'], self.offsets):
            n.attendDig(self.focus_col - lo)
    def shiftCol(self, dir):
        if self.focus_col is not None:
            pos = self.focus_col
            if dir=="prev":
                pos = pos - 1
            elif dir=="next":
                pos = pos + 1
            self.attendCol(pos)

class vProblem(Chunk):
    """ vProblem represents an arithmetic problem written in vertical format """
    def __init__(self, problem=None, features={}):
        super(vProblem, self).__init__({**{
            'type':         'vproblem',
            'operator':     None,
            'operands':     None, 
            'operand1':     None, 
            'operand2':     None,
            'answer':       None,
            'curr_col':     None,
            'curr_ans':     None,
            'this_carry':   None,
            'next_carry':   None,
            'part_prods':   None,
            'prep_zeros':   None}, **features})
        if type(problem) is str:
            p = Problem(problem)
        elif type(problem) is Problem:
            p = problem
        else:
            p = None
        ops = self.features['operands']
        op1 = self.features['operand1']
        op2 = self.features['operand2']
        if p is not None:
            # if problem is given, set other features by problem
            self.features['operator']   = p.features['operator']
            self.features['operands']   = arithMat([p.features[n] for n in ['operand1', 'operand2']])
            self.features['operand1']   = p.features['operand1']
            self.features['operand2']   = p.features['operand2']
            self.features['answer']     = p.features['answer']
        elif isinstance(ops, arithMat):
            # otherwise if operands is given, set operandx by operands
            c = ops.features['contents']
            for i in range(0,len(ops.features['contents'])):
                self.features['operand'+str(i+1)] = c[i]
        elif isinstance(op1, Number) and isinstance(op2, Number):
            # otherwise if operand1 and operand2 are given, set operands by them
            self.features['operands']   = arithMat([op1, op2])
    def __str__(self):
        s = '(vproblem) '
        if self.features['operands'] is not None:
            s = s + str(self.features['operand1']) + " " + self.features['operator'] + " " + str(self.features['operand2']) + " "
            if len(self.features['operands'].features['contents'])>2:
                s = s + self.features['operator'] + " ... "
            s = s + "= "
            if self.features['part_prods'] is not None:
                c = self.features['part_prods'].features['contents']
                s += str([str(n) for n in c]) + " = "
                # s += str(self.features['part_prods']) + " = "
            if self.features['answer'] is None:
                s += "?"
            else:
                s += str(self.features['answer'])
        return(s)
    def alignOperands(self, align, append=False):
        if self.features['operands'] is not None:
            self.features['operands'].alignOperands(align, append)
            self.features['operand1'] = self.features['operands'].features['first']
            self.features['operand2'] = self.features['operands'].features['second']
    def getDecColsFromOperands(self, x=None):
        """ Get number of columns after decimal point in specified operand(s)
            If no operand specified, choose random decimal operand """
        DD = [dd for dd in self.features['operands'].getDecCols()]
        if x is None:
            return(np.random.choice([dd for dd in DD if dd>0]))
        elif x=="max":
            return(max(DD))
        elif x=="operand1":
            return(DD[0])
        elif x=="operand2":
            return(DD[1])
    def setDecDigInAnswer(self, ndd):
        """ Place decimal point in answer so it will have ndd decimal digits """
        dig = self.features['answer'].features['digits']
        if len(dig)<=ndd:
            dig = [0]*(ndd-len(dig)+1) + dig
        wp = [str(i) for i in dig[:(len(dig)-ndd)]]
        dp = [str(i) for i in dig[(len(dig)-ndd):]]
        s = (''.join(wp)) + "." + (''.join(dp))
        self.features['answer'] = Number(s)
    def attendCol(self, pos):
        self.features['operands'].attendCol(pos)
        self.features['curr_col'] = self.features['operands'].features['curr_col']
    def shiftCol(self, dir):
        self.features['operands'].shiftCol(dir)
        self.features['curr_col'] = self.features['operands'].features['curr_col']

class Variable(Chunk):
    def __init__(self, name):
        super(Variable, self).__init__({'type': 'variable', 'name': name})

# Patterns
#   A pattern is a Chunk in which expected features can be None, BLANK, or Variable
#   The basic machinery of production rules (below) relies on matching Chunks to patterns
#   Chunks can be matched to patterns using matchPattern below;
#   doing so will bind values to Variables in a pattern
#   Useful patterns for some Chunk subtypes are provided below

pat_whole       = Number(features={
    'subtype':      'whole'})
pat_negint      = Number(features={
    'subtype':      'negint'})
pat_dec         = Number(features={
    'subtype':      'decimal'})    
pat_SD_whole    = Number(features={
    'subtype':      'whole',
    'multidig':     False})
pat_MD_whole    = Number(features={
    'subtype':      'whole',
    'multidig':     True})
pat_accumulator = Accumulator(features={
    'counter':  Variable('counter'),
    'target':   Variable('target'),
    'total':    Variable('total'),
    'action':   Variable('action'),
    'amount':   Variable('amount')})
pat_fraction    = Number(features={
    'subtype':      'fraction'})
pat_solve_prob  = Goal('solve_problem', {'problem': Variable('problem')})
pat_do_valg     = Goal('do_valg', {
    'algorithm':    Variable('algorithm'),
    'vproblem':     Variable('vproblem'),
    'state':        Variable('state')})
pat_prob        = Problem(features={
    'operator':     Variable('operator'),
    'operand1':     Variable('operand1'),
    'operand2':     Variable('operand2'),
    'answer':       Variable('answer')})
pat_vprob       = vProblem(features={
    'operator':     Variable('operator'),
    'operands':     Variable('operands'), 
    'operand1':     Variable('operand1'), 
    'operand2':     Variable('operand2'),
    'answer':       Variable('answer'),
    'curr_col':     Variable('curr_col'),
    'curr_ans':     Variable('curr_ans'),
    'this_carry':   Variable('this_carry'),
    'next_carry':   Variable('next_carry'),
    'part_prods':   Variable('part_prods'),
    'prep_zeros':   Variable('prep_zeros')})
# pat_lagg: a list to be aggregated (e.g., by adding or multiplying its elements)
pat_lagg        = List(features={
    'length':       Variable('length'),
    'empty':        Variable('empty'),
    'first':        Variable('first'),
    'second':       Variable('second'),
    'last':         Variable('last'),
    'attn_in':      Variable('attn_in'),
    'current':      Variable('current'),
    'list_ans':     Variable('list_ans')})

def matchPattern(pattern, target):
    """
    Given pattern and target, return (match, bindings), 
    where match is a boolean indicating whether target matches pattern, 
    and bindings is a dict of variable bindings used to make the match (or an empty dict if match is False)
    """
    try:
        match       = True
        bindings    = {}
        if not isinstance(pattern, Chunk):
            # If pattern is not a Chunk, target matches pattern iff they are equal or pattern is BLANK and target is None
            match   = (pattern==target) or (pattern=='BLANK' and target is None)
        elif isinstance(pattern, Variable):
            # If pattern is a Variable, target always matches pattern, with binding {pattern name: target}
            bindings = {pattern.features['name']: target}
        elif not isinstance(target, Chunk):
            # If pattern is a Chunk other than a Variable, target must also be a Chunk
            match   = False
        elif len( set([k for k in pattern.features.keys() if pattern.features[k]!='BLANK']) - set(target.features.keys()) ) > 0:
            # If pattern and target are both Chunks, target must have all of pattern's non-blank slots in order to match
            match   = False
        else:
            # In this case, target matches pattern if all shared slots match
            t = target.features
            p = pattern.features
            for slot in p.keys():
                if p[slot] is None:
                    # a shared slot matches without any binding if pattern's value for the slot is None
                    pass
                elif p[slot]=='BLANK' and (slot not in t.keys() or t[slot] is None):
                    # a shared slot matches if pattern's value for the slot is 'BLANK' and target's is None or target lacks the slot
                    # this is a kludgy way to require a slot to be filled with None despite the preceding condition
                    pass
                else:
                    # otherwise, it matches if pattern and target have matching values for that slot
                    # if so, bindings used for the match are added to the dict of variable bindings that will be returned
                    m, b        = matchPattern(p[slot], t[slot])
                    match       = match and m
                    bindings    = {**bindings, **b}
                if not match:
                    bindings = {}
                    break
                    
        return((match, bindings))
    
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in matchPattern:", exc_type, exc_obj, "file", fname, "line", exc_tb.tb_lineno)
        print("Target: ")
        print(str(target))
        print("Pattern: ")
        print(str(pattern))
        sys.exit()

def instantiatePattern(pattern, bindings):
    """
    Replace each Variable in pattern with its corresponding value in bindings
    pattern may be a Chunk, a list, or a dict; bindings is a dict
    """
    if isinstance(pattern, Variable):
        name = pattern.features['name']
        if name in bindings.keys():
            result = bindings[name]
        else:
            result = pattern
    elif isinstance(pattern, Chunk):
        result = pattern
        for slot in result.features.keys():
            result.features[slot] = instantiatePattern(result.features[slot], bindings)
    elif isinstance(pattern, list):
        result = pattern
        for i in range(0,len(result)):
            result[i] = instantiatePattern(result[i], bindings)
    elif isinstance(pattern, dict):
        result = {}
        for key, val in pattern.items():
            result[key] = instantiatePattern(val, bindings)
    else:
        result = pattern
    return(result)

    
### Production Rules

# Rule
#   A class for representing production rules
#   A production rule consists of a condition part and an action part
#   The condition part (which MUST include a goal) specifies when the rule can be used
#   The action part specifies what to do if the rule is used

class Rule:
    def __init__(self, name, goal, conds, acts):
        self.name       = name  # a string
        self.goal       = goal  # a goal
        self.conds      = conds # a list of dicts of form {'target': Chunk, 'pattern': Chunk}
        self.acts       = acts  # a list of dicts of form {'action': str, **params}
    def __str__(self):
        return(self.name)
    def match(self, state, print_mismatch=False):
        """ Attempt to match state to goal pattern and condition patterns 
        If successful, return (match, bindings), otherwise (False, {})"""
        try:
            match, bindings = matchPattern(self.goal, state.getGoal())
            if not match and print_mismatch:
                print("Mismatching goal in Rule.match")
                print(self.goal)
            if match:
                # make current goal and retrieved fact (if any) accessible as variables
                bindings['curr_goal']       = state.getGoal()
                bindings['retrieved_fact']  = state.getRetrievedAnswer()
                if len(self.conds)>0:
                    match, b = matchCondition(['and']+copy.deepcopy(self.conds), bindings, print_mismatch)
                    bindings = {**bindings, **b}
            return((match, bindings))
        except:
            print("Error in Rule.match:", sys.exc_info())
            print("Goal: ")
            print(str(self.goal))
            print("Conds: ")
            for c in self.conds:
                print(str(c))
            sys.exit()
    def fire(self, state, bindings={}):
        acts = copy.deepcopy(self.acts)
        for a in acts:
            for key, val in a.items():
                a[key] = instantiatePattern(val, bindings)
            doPrimitiveAction(a, state, bindings)

# matchCondition determines the kinds of conditions for which Rules can test

def matchCondition(cond, bindings, print_mismatch=False):
    if type(cond) is dict and 'pattern' in cond.keys() and 'target' in cond.keys():
        cond['pattern'] = instantiatePattern(cond['pattern'], bindings)
        cond['target']  = instantiatePattern(cond['target'], bindings)
        m, b            = matchPattern(cond['pattern'], cond['target'])
    elif type(cond) is tuple or type(cond) is list:
        if cond[0]=='and':
            m   = True
            b   = {}
            for i in range(1,len(cond)):
                subcond = cond[i]
                msub, bsub  = matchCondition(subcond, {**bindings, **b}, False)
                if msub:
                    b = {**b, **bsub}
                else:
                    m = False
                    if print_mismatch:
                        print("Mismatch in matchCondition 'and' for condition " + str(i))
                        print(str(subcond))
                        if type(subcond) is dict:
                            print(str(subcond['target']))
                            print(str(subcond['pattern']))
                    break
        elif cond[0]=='or':
            for subcond in cond[1:]:
                msub, bsub  = matchCondition(subcond, bindings, False)
                if msub: break
            m, b = msub, bsub
        elif cond[0]=='not':
            m, b = matchCondition(cond[1], bindings, False)
            m = not m
        elif cond[0] in ['==', '!=', '>=', '<']:
            comparand1  = instantiatePattern(cond[1]['comparand1'], bindings)
            comparand1  = float(0 if comparand1 is None else comparand1)
            comparand2  = instantiatePattern(cond[1]['comparand2'], bindings)
            comparand2  = float(0 if comparand2 is None else comparand2)
            if cond[0]=='==':
                m       = comparand1==comparand2
            elif cond[0]=='!=':
                m       = comparand1!=comparand2
            elif cond[0]=='>=':
                m       = comparand1>=comparand2
            elif cond[0]=='<':
                m       = comparand1<comparand2
            b           = {}
    else:
        print("Unrecognized type for cond in matchCondition: " + str(cond))
    
    if m:
        return(m,b)
    else:
        return(m,{})

    return(m,b)

# doPrimitiveAction determines the kinds of actions that Rules can take when fired

def doPrimitiveAction(a, state, bindings=None):

    try:

        ## Basic Actions
        
        if a['action']=='create_chunk':
            # add chunk to workspace and add variable reference to it to bindings
            if 'copy' in a.keys() and a['copy']==True:
                # create new chunk with copies of passed features,
                # guaranteeing that changes to new chunk won't affect chunks used to create it
                features = {key: copy.deepcopy(val) for key, val in a['features'].items()}
            else:
                features = a['features']
            c = a['chunk_type'](features=features)
            state.addChunk(c)
            bindings[a['variable']] = c
        elif a['action']=='set_feature':
            # set slot in chunk to value
            c = a['chunk']
            s = str(a['slot'])
            if 'value' in a.keys():
                c.addFeatures({s: a['value']})
            elif 'from' in a.keys():
                c.addFeatures({s: a['from'].features[str(a['from_slot'])]})
        elif a['action']=='push_goal':
            # push goal to goal stack
            state.pushGoal(a['goal'])
        elif a['action']=='pop_goal':
            # remove goal from goal stack
            state.popGoal()
        elif a['action']=='deferred_action':
            b = {**{'action': a['def_act']}, **a['features']}
            doPrimitiveAction(b, state, bindings)
        
        ## Actions relating to Attention
        elif a['action']=='attend_dig':
            a['chunk'].attendDig(a['pos'])
        elif a['action']=='shift_dig':
            a['chunk'].shiftDig(a['dir'])
        elif a['action']=='attend_list':
            a['chunk'].attendPos(a['pos'])
        elif a['action']=='shift_list':
            a['chunk'].shiftPos(a['dir'])
        elif a['action']=='attend_col':
            a['chunk'].attendCol(a['pos'])
        elif a['action']=='shift_col':
            a['chunk'].shiftCol(a['dir'])
        
        ## Actions relating to Numbers
        
        elif a['action']=='decompose_whole':
            if a['decomp_type']=='right_digit':
                source  = a['source']
                if source is None or source.features['digits'] is None:
                    targ1   = None
                    targ2   = None
                elif len(source.features['digits'])==1:
                    targ2   = Number(source.features['digits'])
                    targ1   = None
                else:
                    targ2   = Number(source.features['digits'][-1])
                    targ1   = Number(source.features['digits'][0:-1])
                # state.addChunk(targ1)
                # state.addChunk(targ2)
                targ1var, targ2var = a['variables']
                bindings[targ1var] = targ1
                bindings[targ2var] = targ2
        elif a['action']=='prepend_digits':
            if 'number' in a.keys():
                n = a['number']
            elif 'chunk' in a.keys() and a['chunk'].features[a['slot']] is not None:
                n = a['chunk'].features[a['slot']]
            elif 'chunk' in a.keys():
                a['chunk'].features[a['slot']] = Number()
                n = a['chunk'].features[a['slot']]
            n.prependDigits(a['digits'])
        elif a['action']=='append_digits':
            if 'number' in a.keys():
                n = a['number']
            elif 'chunk' in a.keys() and a['chunk'].features[a['slot']] is not None:
                n = a['chunk'].features[a['slot']]
            elif 'chunk' in a.keys():
                a['chunk'].features[a['slot']] = Number()
                n = a['chunk'].features[a['slot']]
            n.appendDigits(a['digits'])

        ## Actions relating to counting

        elif a['action']=='increment':
            # increment chunk or slot in chunk
            c = a['chunk']
            if 'slot' in a.keys() and 'by' in a.keys():
                c.features[a['slot']].increment(a['by'])
            elif 'slot' in a.keys():
                c.features[a['slot']].increment()
            elif 'by' in a.keys():
                c.increment(a['by'])
            else:
                c.increment()
        elif a['action']=='decrement':
            # decrement chunk or slot in chunk
            c = a['chunk']
            if 'slot' in a.keys() and 'by' in a.keys():
                c.features[a['slot']].decrement(a['by'])
            elif 'slot' in a.keys():
                c.features[a['slot']].decrement()
            elif 'by' in a.keys():
                c.decrement(a['by'])
            else:
                c.decrement()

        ## Actions relating to Problems
        
        elif a['action']=='start_problem':
            # TBD: create goal to solve problem and add problem context
            pass
        elif a['action']=='finish_problem':
            # remove goal to solve problem and remove problem context
            state.popGoal()
            
        ## Actions relating to Lists and arithMats
        elif a['action']=='prepend_list':
            if 'list' in a.keys():
                l = a['list']
            elif 'chunk' in a.keys() and a['chunk'].features[a['slot']] is not None:
                l = a['chunk'].features[a['slot']]
            elif 'chunk' in a.keys():
                a['chunk'].features[a['slot']] = List()
                l = a['chunk'].features[a['slot']]
            l.prepend(a['element'])
        elif a['action']=='append_list':
            if 'list' in a.keys():
                l = a['list']
            elif 'chunk' in a.keys() and a['chunk'].features[a['slot']] is not None:
                l = a['chunk'].features[a['slot']]
            elif 'chunk' in a.keys():
                a['chunk'].features[a['slot']] = List()
                l = a['chunk'].features[a['slot']]
            l.append(a['element'])
        elif a['action']=='prepend_arithmat':
            if 'mat' in a.keys():
                m = a['mat']
            elif 'chunk' in a.keys() and a['chunk'].features[a['slot']] is not None:
                m = a['chunk'].features[a['slot']]
            elif 'chunk' in a.keys():
                a['chunk'].features[a['slot']] = arithMat()
                m = a['chunk'].features[a['slot']]
            m.prepend(a['element'])
        elif a['action']=='append_arithmat':
            if 'mat' in a.keys():
                m = a['mat']
            elif 'chunk' in a.keys() and a['chunk'].features[a['slot']] is not None:
                m = a['chunk'].features[a['slot']]
            elif 'chunk' in a.keys():
                a['chunk'].features[a['slot']] = arithMat()
                m = a['chunk'].features[a['slot']]
            m.append(a['element'])
        elif a['action']=='align_operands':
            if 'align' in a.keys() and 'append' in a.keys():
                a['chunk'].alignOperands(a['align'], append=a['append'])
            elif 'align' in a.keys():
                a['chunk'].alignOperands(a['align'])
            else:
                # by default, right-align
                a['chunk'].alignOperands('right')

        ## Actions relating to Decimal Problems in Vertical Format
        elif a['action']=='get_dec_dig_from_op':
            # save number of decimal digits in vproblem operand as variable
            # if 'operand' in a.keys():
                # operand = a['operand']
            # else:
                # operand = 'operand'+str(1+np.random.choice(2))
            operand = a['operand'] if 'operand' in a.keys() else None
            c = Number(a['vproblem'].getDecColsFromOperands(operand))
            state.addChunk(c)
            bindings[a['variable']] = c
        elif a['action']=='set_dec_dig_in_ans':
            if 'dd' in a.keys():
                dd = int(a['dd'])
            elif 'dd_from' in a.keys() and 'dd_slot' in a.keys():
                dd = int(a['dd_from'].features[a['dd_slot']])
            else:
                # by default, decimal point is placed after first digit
                dd = len(a['vproblem'].features['answer'].features['digits']) - 1
            a['vproblem'].setDecDigInAnswer(dd)
            
        ## Actions relating to Fraction Problems
        elif a['action']=='invert_operand':
            p = a['problem']
            o = a['operand'] if 'operand' in a.keys() else np.random.choice(['operand1', 'operand2'])
            n = p.features[o].features['num']
            d = p.features[o].features['den']
            p.features[o] = Number(str(d)+"/"+str(n))
            
        ## Other actions (mostly cheating!)
        
        elif a['action']=='use_calculator':
            p = a['problem']
            x = float(p.features['operand1'])
            y = float(p.features['operand2'])
            if 'lg_by_sm' in a.keys() and a['lg_by_sm']:
                x, y = max([x,y]), min([x,y])
            s = str(x)+p.features['operator']+str(y)
            v = myEval(s)
            if type(v) is float:
                if 'as_int' in a.keys() and a['as_int']:
                    v = int(v)
                else:
                    v = round(v,3)
            p.features['answer'] = Number(str(v))
        elif a['action']=='get_LCM':
            n1 = int(a['number1'])
            n2 = int(a['number2'])
            c = a['chunk']
            s = str(a['slot'])
            c.addFeatures({s: Number(np.lcm(n1,n2))})
        elif a['action']=='get_GCD':
            n1 = int(a['number1'])
            n2 = int(a['number2'])
            c = a['chunk']
            s = str(a['slot'])
            c.addFeatures({s: Number(np.gcd(n1,n2))})
        else:
            pass
            
    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error in doPrimitiveAction:", exc_type, exc_obj, "file", fname, "line", exc_tb.tb_lineno)
        # print("Error in doPrimitiveAction:", sys.exc_info())
        print("Action: ")
        for (key, val) in a.items():
            print(str(key), str(val))
        print("State: ")
        print(str(state))
        print("Bindings: ")
        for b in bindings:
            print(b, bindings[b])
        sys.exit()


### Base Rules (to be included in any UMA model)

R_deferred_action = Rule( # If goal is take deferred action, do so
    name    = 'deferred_action',
    goal    = Goal('deferred_action', {
                'action':   Variable('def_act'),
                'features': Variable('features')}),
    conds   = [],
    acts    = [{'action':   'pop_goal'},
               {'action':   'deferred_action',
                'def_act':  Variable('def_act'),
                'features': Variable('features')}])

R_finish_problem = Rule( # If goal is solve problem that has answer, finish it
    name   = 'finish_problem',
    goal   = Goal('solve_problem', {'problem': Variable('p')}),
    conds  = [# p is either a Problem or a vProblem
              ('or',
               {'target':   Variable('p'),
                'pattern':  pat_prob},
               {'target':   Variable('p'),
                'pattern':  pat_vprob}),
              # answer is not blank
              ('not',
               {'target':   Variable('answer'),
                'pattern':  'BLANK'})],
    acts   = [{'action': 'finish_problem'}])

def make_SFA_acts(operator, operand1, operand2, chunk, slot): # Create problem, solve it, and set chunk slot to its answer
    new_problem_name = 'SFA_problem_' + str(np.random.randint(1,10000))
    return([{'action':       'create_chunk',
             'variable':     new_problem_name,
             'chunk_type':   Problem,
             'features':     {
                'operator':     operator,
                'operand1':     operand1,
                'operand2':     operand2}},
            {'action':   'push_goal',
             'goal':     Goal('deferred_action', {
                'action':   'set_feature',
                'features': {
                    'chunk':        chunk,
                    'slot':         slot,
                    'from':         Variable(new_problem_name),
                    'from_slot':    'answer'}})},
            {'action':       'push_goal',
             'goal':         Goal('solve_problem', {
                'problem':      Variable(new_problem_name)})}])

R_retrieve = Rule( # retrieve answer for + or * with whole operands in [1, 10]
    name    = 'retrieve',
    goal    = pat_solve_prob,
    conds   = [{'target':   Variable('problem'),
                'pattern':  pat_prob},
               {'target':   Variable('answer'),
                'pattern':  'BLANK'},
               # operator is + or *
               ('or',
                {'target':  Variable('operator'),
                 'pattern': '+'},
                {'target':   Variable('operator'),
                 'pattern':  '*'}),
               # operand1 is a whole number in [1,10]
               {'target':  Variable('operand1'),
                'pattern': pat_whole},
               ('<', {
                'comparand1': Variable('operand1'),
                'comparand2': Number(11)}),
               ('not',
                {'target':  Variable('operand1'),
                 'pattern': Number(0)}),
               # operand2 is a whole number in [1,10]
               {'target':  Variable('operand2'),
                'pattern': pat_whole},
               ('<', {
                'comparand1': Variable('operand2'),
                'comparand2': Number(11)}),
               ('not',
                {'target':  Variable('operand2'),
                 'pattern': Number(0)}),
               # a fact has been retrieved
               {'target':   Variable('retrieved_fact'),
                'pattern':  Number()}],
    acts    = [{'action':   'set_feature',
                'chunk':    Variable('problem'),
                'slot':     'answer',
                'value':    Variable('retrieved_fact')}])

default_rules = [R_retrieve, R_finish_problem, R_deferred_action]


### UMA

class UMA:
    def __init__(self, rules=default_rules, answers=mem_answers, params={}):
        ## Parameters
        p = {**{
            # general parameters
            'g':        0.05,   # decision determinism for rules and answers
            'd':        0.4,    # error discount
            'dp':       None,   # error discount for procedure memory
            'da':       None,   # error discount for answer memory
            # answer retrieval parameters
            'c':        4,      # similarity scaling for answer retrieval
            'rt_mu':    7,      # mean of answer retrieval threshold
            'rt_sd':    2,      # standard deviation of answer retrieval threshold
            # features used for procedure memory indices
            'pf':       "['+','-','*',':','whole','FR','MI','WF','ED','UD','WD','DD']"
            }, **params}
        self.g          = p['g']
        self.dp         = (0.5+p['d']) if p['dp'] is None else p['dp']
        self.da         = (0.5+0.5*p['d']) if p['da'] is None else p['da']
        self.c          = p['c']
        self.rt_mu      = p['rt_mu']
        self.rt_sd      = p['rt_sd']
    
        ## Procedural component stores and selects production rules
        self.rules      = [copy.deepcopy(rule) for rule in rules]
        rule_names      = [rule.name for rule in self.rules]
        if len(rule_names)!=len(set(rule_names)):
            print("Error in UMA_mem.__init__: Duplicate names in rules")
            self.rules  = []
            rule_names  = []
        pf              = eval(p['pf'])
        self.proc_mem   = pd.DataFrame(0.0, index=pf, columns=rule_names)
        self.proc_act   = pd.Series(0.0, index=rule_names)
        
        ## Declarative component stores and selects memorized answers
        self.ans_mem    = pd.DataFrame(0.0, index=mem_probs, columns=mem_answers)
        self.prob_sim   = pd.DataFrame(np.exp(-self.c * mem_prob_dists.to_numpy()), index=mem_probs, columns=mem_probs)
                          # similarity exponentially decreases with distance (Kruschke, 1992)
        self.answers    = [Number(answer) for answer in mem_answers]
    
        ## Record of current/most recent model run
        self.trace      = [] # sequence of (state, rule) pairs
        self.runtime    = None
    def __str__(self):
        return("__str__ not implemented for UMA")

    ## Main problem solving loop
    def run(self, input, step_limit=500, learn=False, verbose=True, force_rules=None):
    
        # List names of rules that must be used if possible
        if force_rules is None:
            fnames = None
        elif type(force_rules[0]) is Rule:
            fnames = set([r.name for r in force_rules])
        elif type(force_rules[0]) is str:
            fnames = set(force_rules)
            
        # Initialize variables for production system loop
        if isinstance(input, State):
            state = copy.deepcopy(input)
        else:
            state = State(input)
        stimulate   = True
        context     = state.cs[-1] if len(state.cs)>0 else None
        self.trace  = []
        start       = time.perf_counter()
        
        # Run production system loop
        for step in range(0, step_limit):
            if len(state.gs)>0:
                # Add retrieved fact to state 
                # To save time, only do this if retrieval is potentially relevant
                if state.gs[-1].features['name']=='solve_problem':
                    context = state.getContext()
                    if self.canRetrieve(context):
                        state.ra = self.retrieveAnswer(context)
                        
                try:
                    # Attempt to select a production rule
                    rule, binding = self.chooseRule(state, stimulate, fnames)
                    
                except:
                    # If no rule could be selected, terminate
                    p = str(state.ws[0]) if len(self.trace)==0 else str(self.trace[-1][0].ws[0])
                    if verbose: print("STEP " + str(step) + "\n" + str(state) + "UMA.run while solving " + p + " ended due to rule selection failure\n")
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print("Error in UMA main loop while solving " + p + ": ", exc_type, exc_obj, "file", fname, "line", exc_tb.tb_lineno)
                    self.trace.append((copy.deepcopy(state), None))
                    UMA_traces.append(('Rule selection failure', self.trace))
                    break

                # If no rule matched (chooseRule returned None), terminate gracefully
                if rule is None:
                    p = str(state.ws[0]) if len(self.trace)==0 else str(self.trace[-1][0].ws[0])
                    if verbose: print("STEP " + str(step) + "\n" + str(state) + "UMA.run while solving " + p + " ended due to no matching rules\n")
                    self.trace.append((copy.deepcopy(state), None))
                    UMA_traces.append(('No matching rules', self.trace))
                    break

                # If a rule was selected, fire it
                self.trace.append((copy.deepcopy(state), rule))
                if verbose: print("STEP " + str(step) + "\n" + str(state) + "RULE: " + str(rule) + "\n")
                rule.fire(state, binding)
                    
                # If context changed due to rule firing, clear retrieved fact
                # and record need to stimulate LTM on next run
                new_context = state.cs[-1] if len(state.cs)>0 else None
                if context is None:
                    changed = (new_context is not None)
                elif new_context is None:
                    changed = (context is not None)
                else:
                    changed = context!=new_context
                if changed: state.ra = None
                stimulate   = changed
                context     = new_context
                    
            else:
                # If no goal is present, terminate
                if verbose: print("STEP " + str(step) + "\n" + str(state) + "UMA.run ended due to lack of purpose\n")
                self.trace.append((copy.deepcopy(state), None))
                break
        if verbose and step==step_limit:
            print("UMA.run while solving " + str(self.trace[-1][0].ws[0]) + " reached step limit " + str(step_limit))
        self.runtime = time.perf_counter() - start
        
        # Learning algorithm
        if learn:
        
            # Calculate reward discount due to error, if any
            p       = self.trace[-1][0].ws[0]
            try:
                # Determine whether answer has correct value
                ans     = float(p.features['answer'])
                key     = float(myEval(p.toStrProbOnly()))
                acc     = np.abs(key-ans)<.00001
                if acc and Context(p).data['fraction']==1:
                    # For fraction problems, answers involving decimals are wrong even if value is correct
                    if p.features['answer'].features['subtype']=='fraction':
                        n = p.features['answer'].features['num']
                        d = p.features['answer'].features['den']
                        if n is not None and n.features['subtype']=='decimal':
                            acc = False
                        elif d is not None and d.features['subtype']=='decimal':
                            acc = False
            except:
                # print("UMA.run while solving " + str(p) + " obtained non-floatable answer " + str(p.features['answer']))
                UMA_traces.append(('Non-floatable answer', self.trace))
                acc     = False
            
            # Determine which rules and facts were used
            learn_rules = []
            learn_answers = []
            L = [(state, rule) for state, rule in self.trace if state.getContext() is not None and rule is not None]
            for state, rule in L:
                context     = state.getContext()
                features    = tuple(context.getFeatures())
                learn_rules.append((features, rule.name))
                if rule.name=="retrieve":
                    prob    = context.prob
                    answer  = str(state.ra)
                    learn_answers.append((prob, answer))
                    
            # Treat answers obtained as if they were retrieved from memory
            for p in self.trace[-1][0].ws:
                if type(p) is Problem and p.features['answer'] is not None:
                    context = Context(p)
                    if self.canRetrieve(context):
                        features    = tuple(context.getFeatures())
                        prob        = context.prob
                        learn_rules.append((features, "retrieve"))
                        learn_answers.append((prob, str(p.features['answer'])))

            # Adjust memory for rules
            delta_rule  = 1.0 - (0 if acc else self.dp)
            for features, rule_name in set(learn_rules):
                self.reinforceRule(delta_rule, rule_name, features)
                    
            # Adjust memory for memorized answers
            delta_ans   = 1.0 - (0 if acc else self.da)
            for prob, answer in set(learn_answers):
                self.reinforceAnswer(delta_ans, prob, answer)

        self.learntime = (time.perf_counter() - start) - self.runtime
                    
        # Return trace
        return(self.trace)
    @classmethod
    def printTrace(cls, trace):
        step = 0
        for state, rule in trace:
            print("STEP " + str(step) + "\n" + str(state) + "RULE: " + str(rule) + "\n")
            step = step + 1

    ## General decision rule for procedure memory and answer memory
    def getProbabilities(self, activations):
        probabilities = softmax(activations * self.g)
        return(probabilities)

    ## Procedure memory
    def chooseRule(self, state, stimulate=True, force_names=None):
        """ Identify rules matching state and select by softmax on activations """
        # Find matching rules and bindings for them
        matches     = [(rule.name,)+rule.match(state) for rule in self.rules]
        mnames      = [name for (name, match, binding) in matches if match]
        bindings    = [binding for (name, match, binding) in matches]
        # Calculate probabilities of matching rules
        P = pd.Series(0, index=[rule.name for rule in self.rules])
        if stimulate:
            self.proc_act   = self.getRuleActivations(state.getContext())
        if len(mnames)>0:
            if force_names is None or force_names.isdisjoint(mnames):
                P[mnames]   = self.getProbabilities(self.proc_act[mnames])
            else:
                n           = list(force_names & set(mnames))
                P[n]        = 1.0/len(n)
        # Select a rule and return it together with its binding
        # Normalize P to handle floating point errors in softmax
        if P.sum() > 0:
            P = P / P.sum()
            idx         = np.random.choice(range(0,len(P)), p=P)
            rule        = self.rules[idx]
            binding     = bindings[idx]
            return((rule, binding))
        else:
            # No rules match - return None to signal failure
            return((None, {}))
    def getRuleActivations(self, context):
        C = context.data[self.proc_mem.index]   # features present in context
        A = self.proc_mem.mul(C, axis=0).sum(axis=0) / C.sum()
        return(A)
    def addRule(self, rule):
        if rule.name not in [r.name for r in self.rules]:
            self.rules.append(copy.deepcopy(rule))
            self.proc_mem[rule.name] = 0.0
            self.proc_act[rule.name] = 0.0
    def dropRule(self, rule):
        if rule.name in [r.name for r in self.rules]:
            self.rules = [r for r in self.rules if r.name!=rule.name]
            self.proc_mem.drop(rule.name, axis=1, inplace=True)
            self.proc_act.drop(rule.name, inplace=True)
    def reinforceRule(self, delta, rule, features=None):
        """ Adjust associations of rule with features by delta """
        if type(rule) is str:
            col = rule
        elif type(rule) is Rule:
            col = rule.name
        if col in self.proc_mem.columns:
            if features is None:
                idx = self.proc_mem.index
            else:
                idx = [f for f in features if f in self.proc_mem.index]
            self.proc_mem.loc[idx, col] += delta
    def reinforceAnswer(self, delta, prob, answer):
        if prob in self.ans_mem.index and answer in self.ans_mem.columns:
            self.ans_mem.loc[prob, answer] += delta
    def getRuleProbs(self, problems, rules=[]):
        """ Convenience function to calculate probabilities of rules for problems assuming rule conditions are met """
        D = pd.DataFrame(0.0, index=problems, columns=[r.name for r in rules])
        for p in D.index:
            A           = self.getRuleActivations(Context(p))[D.columns]
            D.loc[p,]   = self.getProbabilities(A)
        return(D)

    ## Answer memory
    def retrieveAnswer(self, context):
        """ Identify memorized answers above rt (retrieval threshold) & select one. rt is on the same scale as activation, and because activation increases by 1 per use (less error discount), rt can be interpreted as "minimum number of previous (correct, maximally efficient) uses needed for answer retrieval". If one or more memorized answers are above rt, one is selected by softmax on their activations. """
        A   = self.getAnswerActivations(context)
        rt  = np.max([1.0, np.random.normal(loc=self.rt_mu, scale=self.rt_sd)])
        R   = A[A>rt]
        if R.shape[0]>0:
            P = pd.Series(0, index=A.index)
            P[R.index] = self.getProbabilities(R)
            # Normalize P to handle floating point errors in softmax
            if P.sum() > 0:
                P = P / P.sum()
            idx = np.random.choice(range(0,len(P)), p=P)
            answer = copy.deepcopy(self.answers[idx])
            return(answer)
        else:
            return(None)
    def getAnswerActivations(self, context):
        S   = self.prob_sim[context.prob]   # similarities of exemplars to problem
        A   = self.ans_mem.mul(S, axis=0).sum(axis=0)
        return(A)
    def canRetrieve(self, context):
        return(context.prob in mem_probs)
    def getRetrievalProbs(self, problems=mem_probs):
        """ Convenience function to estimate probabilities of retrieving answers for problems """
        D = pd.DataFrame(index=problems, columns=['no_retrieval']+list(self.ans_mem.columns))
        for p in D.index:
            """ Estimate probabilities of retrieving answers in context by averaging probabilities from a sample of possible confidence criteria """
            A       = self.getAnswerActivations(Context(p))
            rts     = np.random.normal(loc=self.rt_mu, scale=self.rt_sd, size=30)
            opts    = ['no_retrieval'] + list(self.ans_mem.columns)
            P       = pd.DataFrame(0.0, index=rts, columns=opts)
            for rt in P.index:
                R   = A[A>rt]
                if R.shape[0]>0:
                    P.loc[rt,R.index] = self.getProbabilities(R)
                else:
                    P.loc[rt,'no_retrieval'] = 1.0
            D.loc[p,] = P.mean(axis=0)
        return(D)
    
    ## More convenience functions for inspecting model
    def explainAnswer(self, prob, target, maxIter=100, concise=False, maxLen=50):
        found = False
        for i in range(0,maxIter):
            self.run(State(prob), verbose=False)
            answer  = self.trace[-1][0].ws[0].features['answer']
            if str(target).lower() in ['none', 'blank']:
                # if target is none or blank, any answer that can't be converted to float is a match
                try:
                    x       = float(answer)
                    match   = False
                except:
                    match   = True
            elif str(target)==str(answer):
                # if str(target)==str(answer), then it's a match
                match = True
            else:
                # if neither of the above applies, it's a match iff float(answer)==float(target)
                try:
                    x       = float(answer)
                    y       = float(target)
                    match   = round(x,6)==round(y,6)
                except:
                    match   = False
            if match:
                if concise:
                    trace = [str(rule) for state, rule in self.trace if rule is not None]
                    if len(trace)>maxLen:
                        trace = trace[:int(maxLen/2)] + ["..."] + trace[int(-(maxLen/2)):]
                    for r in trace: print(r)
                else:
                    UMA.printTrace(self.trace)
                break
        if not match:
            print("No explanation found")
    def generateAllAnswers(self, prob, N=200, force_rules=None):
        L = []
        for i in range(N):
            self.run(prob, verbose=False, force_rules=force_rules)
            a = self.trace[-1][0].ws[0].features['answer']
            try:
                L.append(str(a) if a.features['subtype'] in ['fraction','whole'] else str(round(float(a),6)))
            except:
                L.append(str(a))
        return(collections.Counter(L))

def teachRetrieval(M, reps=200):
    """ Teach a model M reliably to retrieve correct answers for all problems where retrieval is applicable """
    for p in mem_probs:
        ans = str(myEval(p))
        if ans in M.ans_mem.columns:
            features    = [f for f in Context(p).getFeatures() if f in M.proc_mem.index]
            M.proc_mem.loc[features, R_retrieve.name] += reps
            M.ans_mem.loc[p, ans] += reps