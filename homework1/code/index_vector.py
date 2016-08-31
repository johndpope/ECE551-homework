# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:46:33 2016

@author: Elad Yarkony
"""

import random

zeros = lambda t: 0. # A function that returns always 0
ones = lambda t: 1. # A function that returns always 1.
rand = lambda t: random.random()


class Vector:
	# Initialize a vector with index set index
	def __init__(self,indices,valfun = zeros):
		self.indices = indices
		self.initWith(valfun)

	# For reading bracket (e.g a=u[i])  operatiorn
	def __getitem__(self,i):
		if (i not in self.indices):
			print("Get error: index", str(i), " not in index set")
		else:
			return self.values[i]
		
	# For writing bracket (e.g u[i]=3)  operatiorn
	def __setitem__(self, i, value): 
		if (i not in self.indices):
			print("Set error: index", str(i), " not in index set")
		else:	
			self.values[i] = value
	
	# String conversion for print calls
	def __str__(self):
		s = str()
		for i in self.indices:
			s+="val["+str(i) + "]=" + str(self[i]) + "\n"
	
		return s

	# Initialize with function val (can use lambdas)
	def initWith(self,val):
		self.values = dict() # Empty dictionary
		for t in self.indices:
			self[t] = val(t)
	
	# Define the "+" operator
	def __add__(self,other):
		if (self.indices!=other.indices):
			print("+ operator: incompatible index types")
			return 0
		
		retv = Vector(self.indices)		
		for t in self.indices:
			retv[t] = self[t]+other[t]
			
		return retv

	# Define the "-" operator
	def __sub__(self,other):
		if (self.indices!=other.indices):
			print("- operator: incompatible index types")
			return 0
		
		retv = Vector(self.indices)		
		for t in self.indices:
			retv[t] = self[t]-other[t]
			
		return retv

	# Define the (right) scalar multiplication 
	def __rmul__(self,scalar):		
		retv = Vector(self.indices)		
		for t in self.indices:
			retv[t] = scalar*self[t]
			
		return retv

	# Define negation (-u)
	def __neg__(self):
		
		retv = Vector(self.indices)		
		for t in self.indices:
			retv[t] = -self[t]
			
		return retv

			
