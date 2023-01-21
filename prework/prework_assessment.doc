

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

	def __init__(self, CustomerID, custname, address, contactdetails):
		self.CustomerID = CustomerID
		self.custname = custname
		self.address = address
		self.contactdetails = contactdetails

	def __str__(self):
		'''
			overwrite str method to print customer details
		'''
		return f'Customer {self.custname}:\tID={self.CustomerID}, address={self.address}, contact={self.contactdetails}'


class Account(Bank):

	def __init__(self, IFSC_Code, bankname, branchname, loc, AccountID, cust, balance):
		super().__init__(IFSC_Code, bankname, branchname, loc)
		self.AccountID = AccountID
		self.cust = cust
		self.balance = balance

	def __str__(self):
		return f'Account {self.AccountID}:\n\tBank: {self.bankname}\n\tCustomer: {self.cust}\n\tBalance: {self.balance}'

	def getAccountInfo(self):
		return f'Account ID {self.AccountID}:\n\tBank {self.bankname}:\tIFSC Code={self.IFSC_Code}, branch={self.branchname}, address={self.loc}\n\t{self.cust}'

	def deposit(self, value, is_valid):
		if is_valid.lower() == 'true':
			self.balance += value
			return f'Deposit Successful:\tNew Balance={self.balance}'
		else:
			return 'Deposit Unsuccessful:\tInvalid Deposit'

	def withdraw(self, value):
		if value > self.balance:
			return 'Withdrawal Unsuccessful:\tNot Enough Funds'
		else:
			self.balance -= value
			return f'Withdrawal Successful:\tNew Balance={self.balance}'

	def getBalance(self):
		return self.balance


class SavingsAccount(Account):

	def __init__(self, IFSC_Code, bankname, branchname, loc, AccountID, cust, balance, SMinBalance):
		super().__init__(IFSC_Code, bankname, branchname, loc, AccountID, cust, balance)
		self.SMinBalance = SMinBalance

	def withdraw(self, value):
		if value > self.balance:
			return 'Withdrawal Unsuccessful:\tNot Enough Funds'
		elif self.balance - value < self.SMinBalance:
			return 'Withdrawal Unsuccessful:\tWithdrawal Below Min Balance'
		else:
			self.balance -= value
			return f'Withdrawal Successful:\tNew Balance={self.balance}'