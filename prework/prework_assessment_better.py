

class Bank(object):

	def __init__(self, IFSC_Code, bankname, branchname, loc):
		self.IFSC_Code = IFSC_Code
		self.bankname = bankname
		self.branchname = branchname
		self.loc = loc

	def __str__(self):
		'''
			overwrite str method to print bank details
		'''
		return f'Bank {self.bankname}:\tIFSC Code={self.IFSC_Code}, branch={self.branchname}, address={self.loc}'


class Customer(object):

	cust_count = 0

	def __init__(self, custname, address, contactdetails):
		Customer.cust_count += 1
		self.CustomerID = Customer.cust_count
		self.custname = custname
		self.address = address
		self.contactdetails = contactdetails

	def __str__(self):
		'''
			overwrite str method to print customer details
		'''
		return f'Customer {self.custname}:\tID={self.CustomerID}, address={self.address}, contact={self.contactdetails}'


class Account(Bank):

	acct_count = 0

	def __init__(self, IFSC_Code, bankname, branchname, loc, cust, balance):
		super().__init__(IFSC_Code, bankname, branchname, loc)
		Account.acct_count += 1
		self.AccountID = Account.acct_count
		self.cust = cust
		self.balance = balance

	def getAccountInfo(self):
		return f'Account ID {self.AccountID}:\n\tBank {self.bankname}:\tIFSC Code={self.IFSC_Code}, branch={self.branchname}, address={self.loc}\n\t{self.cust}'

	def deposit(self, value, is_valid):
		if lower(is_valid) == 'true':
			self.balance += value
			return f'Deposit Successful:\tNew Balance={self.balance}'
		else:
			return 'Deposit Unsuccessful:\tInvalid Deposit'

	def withdraw(self, value):
		if value > self.balance:
			return 'Withdrawal Unsuccessful:\tNot Enough Funds'
		else:
			balance -= value
			return f'Withdrawal Successful:\tNew Balance={self.balance}'

	def getBalance(self):
		return self.balance


class SavingsAccount(Account):

	def __init__(self, IFSC_Code, bankname, branchname, loc, cust, balance, SMinBalance):
		super().__init__(IFSC_Code, bankname, branchname, loc, AccountID, cust, balance)
		self.SMinBalance = SMinBalance

	def withdraw(self, value):
		if value > self.balance:
			return 'Withdrawal Unsuccessful:\tNot Enough Funds'
		elif self.balance - value < self.SMinBalance:
			return 'Withdrawal Unsuccessful:\tWithdrawal Below Min Balance'
		else:
			balance -= value
			return f'Withdrawal Successful:\tNew Balance={self.balance}'