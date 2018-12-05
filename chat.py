"""
Chatbot Client
@author Masha Belyi
"""

import argparse

def main():
	parser = argparse.ArgumentParser(description='Seq2sec model configuration arguments')

	parser.add_argument('--model', type=str, 
    	required=True, help='path to model directory.')
	## 1. load model

	## 2. wait for user input

	## 3. answer

if __name__ == '__main__':
	main()