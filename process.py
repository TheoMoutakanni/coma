# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:39:57 2019

@author: Acer V-NITRO
"""

import os
from facemesh import generateSlicedTimeDataSet

def main():

    if not os.path.exists("spring_female"):
        os.makedirs("spring_female")
    if not os.path.exists("spring_male"):
        os.makedirs("spring_male")
    if not os.path.exists("ceasar"):
        os.makedirs("ceasar")

    print("Preprocessing Data")
    #generateSlicedTimeDataSet("data/spring/SPRING_FEMALE", "spring_female")
    #generateSlicedTimeDataSet("data/spring/SPRING_MALE", "spring_male")
    generateSlicedTimeDataSet("data/caesar-fitted-meshes", "ceasar")

if __name__ == '__main__':
    main()
