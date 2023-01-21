import prework_assessment as pwa



def addBank(bank_list):
	isfc = input('\tInput Bank ISFC Code: ')
	bankname = input('\tInput Bank Name: ')
	branchname = input('\tInput Bank Branch Name: ')
	loc = input('\tInput Bank Location: ')
	bank_list.append(pwa.Bank(isfc, bankname, branchname, loc))
	return 'Bank Added Successfully'


def addCust(cust_list):
	cid = input('\tInput Customer ID: ')
	custname = input('\tInput Customer Name: ')
	addr = input('\tInput Customer Address: ')
	cont = input('\tInput Customer Contact Details: ')
	cust_list.append(pwa.Customer(cid, custname, addr, cont))
	return 'Customer Added Successfully'


def addAcct(acct_list, bank_list, cust_list):
	if not bank_list and not cust_list:
		return 'Must add bank and customer before creating an account'
	elif not bank_list:
		return 'Must add bank before creating an account'
	elif not cust_list:
		return 'Must add customer before creating an account'

	else:
		bank = getArrInput(bank_list, 'Select a bank to create an account with (index above): ')
		cust = getArrInput(cust_list, 'Select a customer creating the account (index above): ')
		ifsc = bank.IFSC_Code
		bankname = bank.bankname
		branch = bank.branchname
		bank_loc = bank.loc
		acct_id = input('\tInput Account ID: ')
		balance = int(input('\tInput Starting Balance: '))
		acct_list.append(pwa.Account(ifsc, bankname, branch, bank_loc, acct_id, cust, balance))
		return 'Account Added Successfully'


def addSAcct(acct_list, bank_list, cust_list):
	if not bank_list and not cust_list:
		return 'Must add bank and customer before creating an account'
	elif not bank_list:
		return 'Must add bank before creating an account'
	elif not cust_list:
		return 'Must add customer before creating an account'

	else:
		bank = getArrInput(bank_list, 'Select a bank to create an account with (index above): ')
		cust = getArrInput(cust_list, 'Select a customer creating the account (index above): ')
		ifsc = bank.IFSC_Code
		bankname = bank.bankname
		branch = bank.branchname
		bank_loc = bank.loc
		acct_id = input('\tInput Account ID: ')
		min_balance = int(input('\tInput Minimum Balance: '))
		balance = int(input('\tInput Starting Balance (>= Min Balance): '))
		acct_list.append(pwa.SavingsAccount(ifsc, bankname, branch, bank_loc, acct_id, cust, balance, min_balance))
		return 'Account Added Successfully'



def getArrInput(array, in_message):
	printArr(array)
	return array[int(input(in_message))]


def printArr(arr):
	print('\n')
	for a in range(len(arr)):
		print(f'{a}:\t{arr[a]}')
	print('\n')



def main():

	menu = 'Welcome to Python Banking! Please refer to the following menu of inputs:\
			\n\tCreate New Bank: new bank\
			\n\tCreate New Customer: new cust\
			\n\tCreate Account (requires a bank & customer): new acct\
			\n\tCreate Savings Account (requires a bank & customer): new sacct\
			\n\n\tPrint Banks: print bank\
			\n\tPrint Customers: print cust\
			\n\tPrint Accounts: print acct\
			\n\n\tDeposit Into Acct: deposit\
			\n\tWithdraw From Acct: withdraw\
			\n\n\tEnd Program: quit\n'

	# banks = [pwa.Bank('AAAA', 'Dunder Mifflin', 'Scranton Branch', 'Sranton, PA'), pwa.Bank('BBBB', 'Doggo Unlimited', 'Sticko Wicko', 'Doggo Park')]
	# customers = [pwa.Customer(654, 'Winnie', '336 2nd Ave', 'Doggo')]
	banks = []
	customers = []
	accounts = []

	print(menu)
	in_val = input('Please Select a Value: ').lower()

	while in_val != 'quit':

		if in_val == 'new bank':
			print(addBank(banks))

		elif in_val == 'new cust':
			print(addCust(customers))

		elif in_val == 'new acct':
			print(addAcct(accounts, banks, customers))

		elif in_val == 'new sacct':
			print(addSAcct(accounts, banks, customers))

		elif in_val == 'print banks':
			printArr(banks)

		elif in_val == 'print cust':
			printArr(customers)

		elif in_val == 'print acct':
			printArr(accounts)

		elif in_val == 'deposit':
			in_acct = getArrInput(accounts, 'Please Select an Account to input into (index above): ')
			in_amt = int(input('How much would you like to deposit: '))
			print(in_acct.deposit(in_amt, 'true'))

		elif in_val == 'withdraw':
			in_acct = getArrInput(accounts, 'Please Select an Account to withdraw from (index above): ')
			in_amt = int(input('How much would you like to withdraw: '))
			print(in_acct.withdraw(in_amt))

		elif in_val == 'quit':
			break

		else:
			print('Invalid Input... Please try again')

		print(f'\n\n{menu}')
		in_val = input('Please Select a Value: ').lower()

	print('\n\nGoodbye')


if __name__ == '__main__':
	main()
