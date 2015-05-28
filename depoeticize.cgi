#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# CGI setup

import cgi

data = cgi.FieldStorage()

text = data.getfirst('text')
model = data.getfirst('model')
error_prob = data.getfirst('errorprob')

print "Content-Type: text/html;charset=utf-8"
print

# Data validation

import re

text = str(text)
model = str(model)
error_prob = float(error_prob)

if len(text) > 2000:
    print "Text must be under 2000 characters!"
    exit()

if model not in ('lm_csr_5k_nvp_2gram', 'lm_csr_5k_nvp_3gram',
                 'lm_csr_20k_nvp_2gram', 'lm_csr_20k_nvp_3gram'):
    print "Invalid language model name!"
    exit()
model = model + '.binary'
if '2gram' in model:
    n = 2
else:
    n = 3

if error_prob <= 0.0 or error_prob >= 1.0:
    print "Probability of typo must be between 0 and 1!"
    exit()

# Load settings

import depoeticizer
depoeticizer.ERROR_PROB = error_prob
depoeticizer.load_language_model(model)

# Depoeticize

text = depoeticizer.depoeticize(text, n=n)
print text.replace('\n', '<br/>').replace(' ', '&nbsp;')
