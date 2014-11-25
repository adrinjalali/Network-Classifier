#!/usr/bin/python
import xml.dom.minidom
import os
import sys
import string
#import re


f=os.popen('qstat -u \* -xml -r')

dom=xml.dom.minidom.parse(f)


jobs=dom.getElementsByTagName('job_info')
run=jobs[0]

runjobs=run.getElementsByTagName('job_list')


def fakeqstat(joblist):
    for r in joblist:
        try:
            jobname=r.getElementsByTagName('JB_name')[0].childNodes[0].data
            jobown=r.getElementsByTagName('JB_owner')[0].childNodes[0].data
            jobstate=r.getElementsByTagName('state')[0].childNodes[0].data
            jobnum=r.getElementsByTagName('JB_job_number')[0].childNodes[0].data
            jobtime='not set'
            if(jobstate=='r'):
                jobtime=r.getElementsByTagName('JAT_start_time')[0].childNodes[0].data
            elif(jobstate=='dt'):
                jobtime=r.getElementsByTagName('JAT_start_time')[0].childNodes[0].data
            else:
                jobtime=r.getElementsByTagName('JB_submission_time')[0].childNodes[0].data

            print(jobnum, '\t', jobown.ljust(16), '\t', jobname.ljust(16),'\t', jobstate,'\t',jobtime)
        except Exception as e:
            print(e)

fakeqstat(runjobs)

