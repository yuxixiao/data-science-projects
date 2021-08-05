#!/usr/bin/env python
# coding: utf-8

# In[96]:


get_ipython().system('pip3 install pandas requests BeautifulSoup4')
get_ipython().system('pip3 install pandas')
get_ipython().system('pip3 install selenium')
get_ipython().system('pip3 install webdriver_manager')


# In[173]:


get_ipython().system('pip3 install lxml')


# In[128]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
import selenium
import sys
import time
#import http.client
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.firefox.options import Options
from datetime import datetime as dt
from webdriver_manager.firefox import GeckoDriverManager


# In[106]:


def driverCreate():
  print("Creating Driver")
  options = Options()
  options.headless = True #This option disables the browser window on the screen
  #profile = webdriver.SafariProfile(r'/usr/bin/safaridriver')
  driver = webdriver.Safari(executable_path='/usr/bin/safaridriver')
  return driver
def sleep(x, driver):
  for y in range(x, -1, -1):
    sys.stdout.write('\r')
    sys.stdout.write('{:2d} seconds'.format(y))
    sys.stdout.flush()
    time.sleep(1)
  sys.stdout.write('\r')


# In[117]:


if __name__== '__main__':
  #url = "https://www.tacobell.ca/en/nutrition/"
  url = "https://restaurant.nutritionix.com/taco-bell-canada/landing?lang=en#vegetarian-35991"
  driver = driverCreate()
  driver.get(url)
  sleep(25, driver)
  #iframe = driver.find_element_by_css_selector("[title*='Menu nutritionnel interactif']")
  #driver.switch_to_frame(iframe)
  html = driver.page_source
  soup = bs(html, 'html.parser')
  print("Soup built, looping through div")
  #Loop through all the div and print the Text, this will show the items as seen on snippet sent over, only pick the menu items
  for item in soup.find_all('div',attrs={'class':'item-name'}):
    print(item.text)
  for category in soup.find_all('div',attrs={'class':'list-item-content'}):  #, attrs={'class':'category5'}
    print(category.text)
  print("Loop completed, goodbye!")
  driver.quit()


# In[212]:


conf = pd.read_csv('/Users/XD/Desktop/config.csv')
#Loop to go over all pages
df = {}
x = conf['id']


# In[226]:


base_url = "https://restaurant.nutritionix.com/taco-bell-canada/item/{0}"
end_url = "?lang=en"
for i in range(len(x)):
     urls = base_url.format(x[i])+end_url
     if __name__== '__main__':
      driver = driverCreate()
      driver.get(urls)
      sleep(25, driver)
      html = driver.page_source
      soup = bs(html, 'html.parser')
      print("Soup built, looping through div")
      #Loop through all the div and print the Text, this will show the items as seen on snippet sent over, only pick the menu items
      for item in soup.find_all('ul',attrs={'class':'list-review-items'}):
        print(item.text)
      print("Loop completed, goodbye!")
      driver.quit()


# In[230]:


base_url = "https://restaurant.nutritionix.com/taco-bell-canada/item/{0}"
end_url = "?lang=en"
for i in range(len(x)):
     urls = base_url.format(x[i])+end_url
     if __name__== '__main__':
      driver = driverCreate()
      driver.get(urls)
      sleep(25, driver)
      html = driver.page_source
      soup = bs(html, 'html.parser')
      print("Soup built, looping through div")
      #Loop through all the div and print the Text, this will show the items as seen on snippet sent over, only pick the menu items
      for item in soup.find_all('div',attrs={'class':'nf-calories'}):
        print(item.text)
      print("Loop completed, goodbye!")
      driver.quit()


# In[78]:


response = requests.get(url)
soup = bs(response.content, 'html.parser') 
soup

