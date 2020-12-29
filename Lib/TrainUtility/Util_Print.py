from typing import *


class UtilPrint_Component:

	def __init__(self):
		super().__init__()

		# data
		# ...

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def content(self):
		raise NotImplementedError

	# Operation
	def getContent(self) -> str:
		raise NotImplementedError


class UtilPrint_Level:

	def __init__(self):
		super().__init__()

		# data
		self.indent: 	int = 4
		self.level_cur: int = 0
		self.level_max: int = 20

		# operation
		# ...

	def __del__(self):
		return

	# Operation
	def createIndent(self) -> str:
		content: str = " " * self.indent * self.level_cur
		return content

	def incrementLevel(self) -> None:
		self.level_cur = min(self.level_cur + 1, self.level_max)

	def decrementLevel(self) -> None:
		self.level_cur = max(self.level_cur - 1, 0)

	def resetLevel(self) -> None:
		self.level_cur = 0


class UtilPrint_Table:

	"""
	- vertical / column line: 		separate each column
	- horizontal / separation line: separate title and item
	- padding inner:				pad content box
	- padding outer:				pad table box
	"""

	def __init__(self):
		super().__init__()

		# data
		self.padding_inner:	int = 1
		self.padding_outer: int = 1

		self._title_list: 	List[str] 		= []
		self._item_list:	List[List[str]] = []

		# operation
		# ...

	def __del__(self):
		return

	# Property
	@property
	def title_list(self) -> List[str]:
		return self._title_list.copy()

	@property
	def item_list(self) -> List[List[str]]:
		return self._item_list.copy()

	# Operation
	def createTable(self, config_level: UtilPrint_Level) -> str:
		# indent
		indent: str = ''
		if config_level is not None:
			indent = config_level.createIndent()

		# ----- get length max -----
		# first get the max string of item and title in each column
		# assume: len(title_list) == len(item_list)
		length_list: List[int] = [0 for _ in range(len(self._title_list))]

		for index in range(len(self._title_list)):

			len_max: int = 0

			# title length
			len_max = max(len_max, len(self._title_list[index]))

			# item length
			for string_list in self._item_list:
				len_max = max(len_max, len(string_list[index]))

			length_list[index] = len_max

		# ----- draw the table -----
		content: str = ""

		# title
		content += indent
		content += self._fillRow_(self._title_list, length_list)
		content += '\n'

		# separation line
		# length_sum:		sum of length in length_list
		# length_line:		sum of vertical line (column line)
		# length_padding 	sum of padding space
		line: str = ""
		for index, length in enumerate(length_list):

			# intersection of separation line and column line
			if index != 0:
				line += '+'

			# padding outer
			if index == 0:
				line += '-' * self.padding_outer
			elif index == len(length_list) - 1:
				line += '-' * self.padding_outer

			# separation line
			line += '-' * (length + self.padding_inner * 2)

		content += indent
		content += line
		content += '\n'

		# item
		for string_list in self._item_list:
			content += indent
			content += self._fillRow_(string_list, length_list)
			content += '\n'

		return content

	# column
	def addColumn(self, title: str) -> None:
		# add title to title list
		# then append one empty string to each item of the item_list
		self._title_list.append(title)

		for item in self._item_list:
			item.append("")

	# backup
	# def rmColumn(self, index: int) -> None:
	# 	pass

	# row
	def addRow(self, item_list: List[str]) -> None:
		# check if len(item_list) == len(self.title_list)
		# if same, then ok
		# if item_list is too long, then remove the last few items
		# if item_list is too short, then append it
		if len(item_list) == len(self._title_list):
			self._item_list.append(item_list)

		elif len(item_list) > len(self._title_list):
			self._item_list.append(item_list[:-(len(item_list) - len(self._item_list))])

		else:
			item_list.extend(["" for _ in range(len(self._item_list) - len(item_list))])
			self._item_list.append(item_list)

	def extendRow(self, row_list: List[List[str]]) -> None:
		for row in row_list:
			self.addRow(row)

	# backup
	# def rmRow(self, index: int) -> None:
	# 	pass

	def resetRow(self) -> None:
		self._item_list.clear()

	# Protected
	def _fillRow_(self, string_list: List[str], length_list: List[int]) -> str:
		line: str = ""
		for index, item in enumerate(string_list):

			# column line
			if index != 0:
				line += '|'

			# padding outer
			if index == 0:
				line += ' ' * self.padding_outer
			elif index == len(string_list) - 1:
				line += ' ' * self.padding_outer

			# padding (inner - left)
			line += ' ' * self.padding_inner

			# add title
			line += item

			# add space to fix the box size
			line += ' ' * (length_list[index] - len(item))

			# padding (inner - right)
			line += ' ' * self.padding_inner

		return line


if __name__ == '__main__':
	# indent
	util_level = UtilPrint_Level()
	util_level.incrementLevel()

	# table
	util_table = UtilPrint_Table()
	util_table.padding_inner = 5

	util_table.addColumn("Column 1")
	util_table.addColumn("Column 2")
	util_table.addColumn("Column 3")

	util_table.addRow(["00", "01", "02"])
	util_table.addRow(["10", "11", "12"])
	util_table.addRow(["20", "21", "22"])

	content_: str = util_table.createTable(util_level)
	print(content_)
