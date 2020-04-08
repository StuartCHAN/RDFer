# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 20:36:33 2020

@author: Stuart
"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points, model_name, stamp):
    r"""Showing and saving plots.

    Inputs:.
        - points: loss valus (a list of floats)
        - model_name: string
        - stamp: a string from struc UTC-time
    """
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    #plt.plot(losses)
    plt.plot(points)
    try:
        plt.savefig('./dataset/model/%(model_name)s/%(stamp)s/%(stamp)s.png'%{ "model_name":model_name, "stamp":stamp } )
    except:
        plt.savefig('./dataset/model/%(model_name)s/%(stamp)s.png'%{ "model_name":model_name, "stamp":stamp } )
