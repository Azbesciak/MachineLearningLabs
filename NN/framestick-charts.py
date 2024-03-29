import math
import matplotlib.pyplot as plt
import numpy as np

from NN_helpers import write_pdf

charts = [
    """Script.Message: 0 9 17
Script.Message: 1 9 18
Script.Message: 2 10 18
Script.Message: 3 10 19
Script.Message: 4 9 15
Script.Message: 5 10 15
Script.Message: 6 10 14
Script.Message: 7 9 9
Script.Message: 8 10 9
Script.Message: 9 11 9
Script.Message: 10 10 8
Script.Message: 11 10 9
Script.Message: 12 10 10
Script.Message: 13 10 11
Script.Message: 14 9 8
Script.Message: 15 9 9
Script.Message: 16 9 8
Script.Message: 17 9 9
Script.Message: 18 8 8
Script.Message: 19 9 8
Script.Message: 20 8 8
Script.Message: 21 8 9
Script.Message: 22 8 8
Script.Message: 23 8 9
Script.Message: 24 8 8
Script.Message: 25 9 8
Script.Message: 26 10 8
Script.Message: 27 10 9
Script.Message: 28 10 10
Script.Message: 29 10 9
Script.Message: 30 9 6
Script.Message: 31 9 5
Script.Message: 32 10 5
Script.Message: 33 11 5
Script.Message: 34 12 5
Script.Message: 35 12 4
Script.Message: 36 13 4
Script.Message: 37 13 5
Script.Message: 38 13 6
Script.Message: 39 13 5
Script.Message: 40 14 5
Script.Message: 41 13 5
Script.Message: 42 13 4
Script.Message: 43 13 5
Script.Message: 44 13 4
Script.Message: 45 12 3
Script.Message: 46 12 4
Script.Message: 47 11 4
Script.Message: 48 11 5
Script.Message: 49 11 6
Script.Message: 50 11 7
Script.Message: 51 10 6
Script.Message: 52 10 7
Script.Message: 53 10 6
Script.Message: 54 10 5
Script.Message: 55 11 5
Script.Message: 56 10 3
Script.Message: 57 10 2
Script.Message: 58 10 1
Script.Message: 59 10 2
Script.Message: 60 10 3
Script.Message: 61 11 3
Script.Message: 62 12 3
Script.Message: 63 13 3
Script.Message: 64 12 3
Script.Message: 65 12 2
Script.Message: 66 12 3
Script.Message: 67 13 3
Script.Message: 68 14 3
Script.Message: 69 14 2
Script.Message: 70 13 2
Script.Message: 71 13 3
Script.Message: 72 14 3
Script.Message: 73 14 4
Script.Message: 74 14 3
Script.Message: 75 13 1
Script.Message: 76 13 0
Script.Message: 77 14 0
Script.Message: 78 15 0
Script.Message: 79 16 0
Script.Message: 80 15 0
Script.Message: 81 15 1
Script.Message: 82 15 0
Script.Message: 83 14 0
Script.Message: 84 14 1
Script.Message: 85 13 1
Script.Message: 86 13 2
Script.Message: 87 13 3
Script.Message: 88 12 2
Script.Message: 89 13 2
Script.Message: 90 12 2
Script.Message: 91 11 2
Script.Message: 92 11 3
Script.Message: 93 12 3
Script.Message: 94 12 2
Script.Message: 95 13 2
Script.Message: 96 12 2
Script.Message: 97 12 3
Script.Message: 98 11 3
Script.Message: 99 10 2
Script.Message: 100 9 1
Script.Message: 101 8 1
Script.Message: 102 8 0
Script.Message: 103 9 0
Script.Message: 104 10 0
Script.Message: 105 9 0
Script.Message: 106 9 1
Script.Message: 107 9 0
Script.Message: 108 10 0
Script.Message: 109 9 0
Script.Message: 110 9 1
Script.Message: 111 9 2
Script.Message: 112 10 2
Script.Message: 113 11 2
Script.Message: 114 11 3
Script.Message: 115 11 2
Script.Message: 116 11 3
Script.Message: 117 10 2
Script.Message: 118 9 2
Script.Message: 119 9 3
Script.Message: 120 9 4
Script.Message: 121 9 3
Script.Message: 122 9 4
Script.Message: 123 9 5
Script.Message: 124 10 5
Script.Message: 125 10 4
Script.Message: 126 9 4
Script.Message: 127 8 3
Script.Message: 128 8 4
Script.Message: 129 8 3
Script.Message: 130 8 2
Script.Message: 131 7 2
Script.Message: 132 6 1
Script.Message: 133 6 0
Script.Message: 134 5 0
Script.Message: 135 6 0
Script.Message: 136 6 1
Script.Message: 137 5 1
Script.Message: 138 5 2
Script.Message: 139 5 3
Script.Message: 140 4 2
Script.Message: 141 3 1
Script.Message: 142 3 0
Script.Message: 143 2 0
Script.Message: 144 2 1
Script.Message: 145 2 0
Script.Message: 146 1 0
Script.Message: 147 2 0
Script.Message: 148 3 0
Script.Message: 149 4 0
Script.Message: 150 3 0
Script.Message: 151 4 0
Script.Message: 152 5 0
Script.Message: 153 5 1
Script.Message: 154 5 2
Script.Message: 155 4 2
Script.Message: 156 3 0
Script.Message: 157 3 1
Script.Message: 158 3 2
Script.Message: 159 2 2
Script.Message: 160 3 2
Script.Message: 161 2 1
Script.Message: 162 2 0
Script.Message: 163 1 0
Script.Message: 164 1 1
Script.Message: 165 2 1
Script.Message: 166 3 1
Script.Message: 167 2 0
Script.Message: 168 1 0
Script.Message: 169 2 0
Script.Message: 170 2 1
Script.Message: 171 2 2
Script.Message: 172 2 1
Script.Message: 173 2 0
Script.Message: 174 3 0
Script.Message: 175 4 0
Script.Message: 176 3 0
Script.Message: 177 3 1
Script.Message: 178 2 0
Script.Message: 179 2 1
Script.Message: 180 2 2
Script.Message: 181 2 3
Script.Message: 182 2 4
Script.Message: 183 3 4
Script.Message: 184 3 5
Script.Message: 185 3 6
Script.Message: 186 3 5
Script.Message: 187 2 1
Script.Message: 188 1 0
Script.Message: 189 1 1
Script.Message: 190 2 1
Script.Message: 191 1 0
Script.Message: 192 1 1
Script.Message: 193 0 0
Script.Message: 194 1 0
Script.Message: 195 2 0
Script.Message: 196 1 0
Script.Message: 197 1 1
Script.Message: 198 1 0
Script.Message: 199 1 1
    """,
    """Script.Message: 0 10 18
Script.Message: 1 10 17
Script.Message: 2 10 16
Script.Message: 3 11 16
Script.Message: 4 11 17
Script.Message: 5 11 18
Script.Message: 6 11 19
Script.Message: 7 11 18
Script.Message: 8 11 17
Script.Message: 9 12 17
Script.Message: 10 11 15
Script.Message: 11 11 14
Script.Message: 12 11 15
Script.Message: 13 10 12
Script.Message: 14 9 11
Script.Message: 15 10 11
Script.Message: 16 9 10
Script.Message: 17 9 11
Script.Message: 18 8 6
Script.Message: 19 7 3
Script.Message: 20 7 2
Script.Message: 21 8 2
Script.Message: 22 8 3
Script.Message: 23 9 3
Script.Message: 24 8 1
Script.Message: 25 9 1
Script.Message: 26 10 1
Script.Message: 27 10 2
Script.Message: 28 10 3
Script.Message: 29 11 3
Script.Message: 30 11 2
Script.Message: 31 11 3
Script.Message: 32 10 2
Script.Message: 33 10 3
Script.Message: 34 9 3
Script.Message: 35 9 4
Script.Message: 36 8 4
Script.Message: 37 7 4
Script.Message: 38 7 5
Script.Message: 39 8 5
Script.Message: 40 8 4
Script.Message: 41 7 3
Script.Message: 42 7 2
Script.Message: 43 7 3
Script.Message: 44 8 3
Script.Message: 45 8 4
Script.Message: 46 8 3
Script.Message: 47 8 2
Script.Message: 48 7 2
Script.Message: 49 6 2
Script.Message: 50 6 1
Script.Message: 51 6 0
Script.Message: 52 7 0
Script.Message: 53 8 0
Script.Message: 54 9 0
Script.Message: 55 8 0
Script.Message: 56 7 0
Script.Message: 57 7 1
Script.Message: 58 7 0
Script.Message: 59 8 0
Script.Message: 60 7 0
Script.Message: 61 7 1
Script.Message: 62 8 1
Script.Message: 63 8 0
Script.Message: 64 8 1
Script.Message: 65 8 2
Script.Message: 66 8 3
Script.Message: 67 9 3
Script.Message: 68 8 2
Script.Message: 69 8 1
Script.Message: 70 9 1
Script.Message: 71 10 1
Script.Message: 72 10 0
Script.Message: 73 9 0
Script.Message: 74 8 0
Script.Message: 75 7 0
Script.Message: 76 6 0
Script.Message: 77 6 1
Script.Message: 78 7 1
Script.Message: 79 7 0
Script.Message: 80 8 0
Script.Message: 81 7 0
Script.Message: 82 7 1
Script.Message: 83 7 2
Script.Message: 84 7 3
Script.Message: 85 7 4
Script.Message: 86 7 3
Script.Message: 87 6 3
Script.Message: 88 6 2
Script.Message: 89 7 2
Script.Message: 90 8 2
Script.Message: 91 8 1
Script.Message: 92 9 1
Script.Message: 93 9 2
Script.Message: 94 9 1
Script.Message: 95 9 0
Script.Message: 96 9 1
Script.Message: 97 9 0
Script.Message: 98 9 1
Script.Message: 99 9 2
Script.Message: 100 9 3
Script.Message: 101 9 2
Script.Message: 102 9 1
Script.Message: 103 9 0
Script.Message: 104 8 0
Script.Message: 105 7 0
Script.Message: 106 8 0
Script.Message: 107 9 0
Script.Message: 108 8 0
Script.Message: 109 7 0
Script.Message: 110 8 0
Script.Message: 111 7 0
Script.Message: 112 7 1
Script.Message: 113 8 1
Script.Message: 114 8 0
Script.Message: 115 9 0
Script.Message: 116 10 0
Script.Message: 117 9 0
Script.Message: 118 8 0
Script.Message: 119 8 1
Script.Message: 120 8 2
Script.Message: 121 9 2
Script.Message: 122 9 3
Script.Message: 123 8 2
Script.Message: 124 7 2
Script.Message: 125 6 2
Script.Message: 126 6 3
Script.Message: 127 5 3
Script.Message: 128 5 4
Script.Message: 129 4 4
Script.Message: 130 5 4
Script.Message: 131 5 3
Script.Message: 132 6 3
Script.Message: 133 6 4
Script.Message: 134 5 3
Script.Message: 135 5 4
Script.Message: 136 6 4
Script.Message: 137 6 5
Script.Message: 138 7 5
Script.Message: 139 8 5
Script.Message: 140 7 3
Script.Message: 141 7 4
Script.Message: 142 6 3
Script.Message: 143 7 3
Script.Message: 144 6 1
Script.Message: 145 5 0
Script.Message: 146 6 0
Script.Message: 147 7 0
Script.Message: 148 8 0
Script.Message: 149 8 1
Script.Message: 150 8 2
Script.Message: 151 9 2
Script.Message: 152 10 2
Script.Message: 153 9 1
Script.Message: 154 9 2
Script.Message: 155 10 2
Script.Message: 156 10 1
Script.Message: 157 10 0
Script.Message: 158 9 0
Script.Message: 159 10 0
Script.Message: 160 10 1
Script.Message: 161 10 2
Script.Message: 162 10 3
Script.Message: 163 10 4
Script.Message: 164 10 5
Script.Message: 165 10 4
Script.Message: 166 9 3
Script.Message: 167 8 3
Script.Message: 168 9 3
Script.Message: 169 8 2
Script.Message: 170 8 3
Script.Message: 171 7 2
Script.Message: 172 7 3
Script.Message: 173 7 4
Script.Message: 174 7 3
Script.Message: 175 8 3
Script.Message: 176 9 3
Script.Message: 177 8 3
Script.Message: 178 9 3
Script.Message: 179 8 3
Script.Message: 180 9 3
Script.Message: 181 9 4
Script.Message: 182 9 5
Script.Message: 183 8 5
Script.Message: 184 8 4
Script.Message: 185 8 5
Script.Message: 186 8 4
Script.Message: 187 8 5
Script.Message: 188 8 6
Script.Message: 189 8 7
Script.Message: 190 9 7
Script.Message: 191 9 8
Script.Message: 192 10 8
Script.Message: 193 10 7
Script.Message: 194 10 6
Script.Message: 195 9 6
Script.Message: 196 9 5
Script.Message: 197 9 4
Script.Message: 198 9 5
Script.Message: 199 9 4
    """,
    """Script.Message: 0 9 17
Script.Message: 1 9 18
Script.Message: 2 9 17
Script.Message: 3 8 15
Script.Message: 4 8 16
Script.Message: 5 8 15
Script.Message: 6 8 14
Script.Message: 7 8 15
Script.Message: 8 9 15
Script.Message: 9 10 15
Script.Message: 10 10 14
Script.Message: 11 11 14
Script.Message: 12 11 13
Script.Message: 13 11 12
Script.Message: 14 10 9
Script.Message: 15 9 9
Script.Message: 16 9 10
Script.Message: 17 9 11
Script.Message: 18 9 12
Script.Message: 19 9 13
Script.Message: 20 9 12
Script.Message: 21 9 13
Script.Message: 22 10 13
Script.Message: 23 11 13
Script.Message: 24 10 7
Script.Message: 25 10 6
Script.Message: 26 10 7
Script.Message: 27 10 8
Script.Message: 28 10 7
Script.Message: 29 10 6
Script.Message: 30 10 5
Script.Message: 31 10 6
Script.Message: 32 9 5
Script.Message: 33 9 4
Script.Message: 34 10 4
Script.Message: 35 10 3
Script.Message: 36 11 3
Script.Message: 37 10 3
Script.Message: 38 9 3
Script.Message: 39 8 2
Script.Message: 40 8 3
Script.Message: 41 8 4
Script.Message: 42 9 4
Script.Message: 43 8 4
Script.Message: 44 9 4
Script.Message: 45 9 5
Script.Message: 46 9 4
Script.Message: 47 8 2
Script.Message: 48 8 3
Script.Message: 49 8 4
Script.Message: 50 7 3
Script.Message: 51 7 2
Script.Message: 52 8 2
Script.Message: 53 7 1
Script.Message: 54 6 1
Script.Message: 55 7 1
Script.Message: 56 7 2
Script.Message: 57 6 2
Script.Message: 58 6 3
Script.Message: 59 5 1
Script.Message: 60 5 2
Script.Message: 61 5 1
Script.Message: 62 4 1
Script.Message: 63 4 0
Script.Message: 64 3 0
Script.Message: 65 4 0
Script.Message: 66 3 0
Script.Message: 67 3 1
Script.Message: 68 2 0
Script.Message: 69 3 0
Script.Message: 70 3 1
Script.Message: 71 3 2
Script.Message: 72 3 3
Script.Message: 73 3 2
Script.Message: 74 4 2
Script.Message: 75 4 1
Script.Message: 76 4 0
Script.Message: 77 3 0
Script.Message: 78 4 0
Script.Message: 79 5 0
Script.Message: 80 4 0
Script.Message: 81 3 0
Script.Message: 82 4 0
Script.Message: 83 3 0
Script.Message: 84 3 1
Script.Message: 85 3 2
Script.Message: 86 3 3
Script.Message: 87 3 2
Script.Message: 88 2 0
Script.Message: 89 3 0
Script.Message: 90 4 0
Script.Message: 91 5 0
Script.Message: 92 4 0
Script.Message: 93 5 0
Script.Message: 94 5 1
Script.Message: 95 4 1
Script.Message: 96 3 0
Script.Message: 97 2 0
Script.Message: 98 1 0
Script.Message: 99 2 0
Script.Message: 100 3 0
Script.Message: 101 3 1
Script.Message: 102 2 0
Script.Message: 103 3 0
Script.Message: 104 4 0
Script.Message: 105 4 1
Script.Message: 106 3 0
Script.Message: 107 3 1
Script.Message: 108 2 0
Script.Message: 109 1 0
Script.Message: 110 0 0
Script.Message: 111 1 0
Script.Message: 112 1 1
Script.Message: 113 1 0
Script.Message: 114 1 1
Script.Message: 115 0 0
Script.Message: 116 1 0
Script.Message: 117 1 1
Script.Message: 118 0 0
Script.Message: 119 1 0
Script.Message: 120 2 0
Script.Message: 121 1 0
Script.Message: 122 2 0
Script.Message: 123 3 0
Script.Message: 124 3 1
Script.Message: 125 2 0
Script.Message: 126 1 0
Script.Message: 127 1 1
Script.Message: 128 0 0
Script.Message: 129 1 0
Script.Message: 130 1 1
Script.Message: 131 0 0
Script.Message: 132 1 0
Script.Message: 133 1 1
Script.Message: 134 2 1
Script.Message: 135 2 2
Script.Message: 136 1 1
Script.Message: 137 1 0
Script.Message: 138 0 0
Script.Message: 139 1 0
Script.Message: 140 2 0
Script.Message: 141 1 0
Script.Message: 142 2 0
Script.Message: 143 3 0
Script.Message: 144 2 0
Script.Message: 145 2 1
Script.Message: 146 1 0
Script.Message: 147 1 1
Script.Message: 148 2 1
Script.Message: 149 1 0
Script.Message: 150 0 0
Script.Message: 151 1 0
Script.Message: 152 0 0
Script.Message: 153 1 0
Script.Message: 154 1 1
Script.Message: 155 0 0
Script.Message: 156 1 0
Script.Message: 157 0 0
Script.Message: 158 1 0
Script.Message: 159 2 0
Script.Message: 160 2 1
Script.Message: 161 1 1
Script.Message: 162 1 0
Script.Message: 163 2 0
Script.Message: 164 2 1
Script.Message: 165 3 1
Script.Message: 166 4 1
Script.Message: 167 4 0
Script.Message: 168 5 0
Script.Message: 169 5 1
Script.Message: 170 6 1
Script.Message: 171 5 1
Script.Message: 172 6 1
Script.Message: 173 6 0
Script.Message: 174 7 0
Script.Message: 175 7 1
Script.Message: 176 7 0
Script.Message: 177 6 0
Script.Message: 178 6 1
Script.Message: 179 6 2
Script.Message: 180 6 3
Script.Message: 181 6 4
Script.Message: 182 6 3
Script.Message: 183 7 3
Script.Message: 184 6 2
Script.Message: 185 5 2
Script.Message: 186 5 3
Script.Message: 187 4 3
Script.Message: 188 3 0
Script.Message: 189 3 1
Script.Message: 190 2 0
Script.Message: 191 1 0
Script.Message: 192 2 0
Script.Message: 193 2 1
Script.Message: 194 3 1
Script.Message: 195 2 0
Script.Message: 196 2 1
Script.Message: 197 2 2
Script.Message: 198 3 2
Script.Message: 199 3 1
    """,
    """
Script.Message: 0 9 19
Script.Message: 1 8 16
Script.Message: 2 9 16
Script.Message: 3 9 17
Script.Message: 4 10 17
Script.Message: 5 10 18
Script.Message: 6 10 17
Script.Message: 7 10 18
Script.Message: 8 10 17
Script.Message: 9 9 14
Script.Message: 10 9 15
Script.Message: 11 10 15
Script.Message: 12 11 15
Script.Message: 13 11 14
Script.Message: 14 11 13
Script.Message: 15 11 12
Script.Message: 16 11 11
Script.Message: 17 12 11
Script.Message: 18 12 10
Script.Message: 19 12 11
Script.Message: 20 13 11
Script.Message: 21 12 9
Script.Message: 22 12 10
Script.Message: 23 13 10
Script.Message: 24 13 9
Script.Message: 25 13 8
Script.Message: 26 12 8
Script.Message: 27 13 8
Script.Message: 28 12 6
Script.Message: 29 13 6
Script.Message: 30 13 7
Script.Message: 31 13 6
Script.Message: 32 13 5
Script.Message: 33 14 5
Script.Message: 34 14 6
Script.Message: 35 14 7
Script.Message: 36 14 6
Script.Message: 37 14 5
Script.Message: 38 14 4
Script.Message: 39 15 4
Script.Message: 40 15 5
Script.Message: 41 14 5
Script.Message: 42 15 5
Script.Message: 43 15 6
Script.Message: 44 16 6
Script.Message: 45 16 5
Script.Message: 46 16 6
Script.Message: 47 17 6
Script.Message: 48 17 5
Script.Message: 49 17 4
Script.Message: 50 17 5
Script.Message: 51 17 4
Script.Message: 52 16 4
Script.Message: 53 16 3
Script.Message: 54 17 3
Script.Message: 55 16 2
Script.Message: 56 16 1
Script.Message: 57 17 1
Script.Message: 58 16 1
Script.Message: 59 16 0
Script.Message: 60 15 0
Script.Message: 61 14 0
Script.Message: 62 14 1
Script.Message: 63 14 2
Script.Message: 64 15 2
Script.Message: 65 15 3
Script.Message: 66 15 4
Script.Message: 67 15 3
Script.Message: 68 15 2
Script.Message: 69 16 2
Script.Message: 70 15 1
Script.Message: 71 15 2
Script.Message: 72 15 1
Script.Message: 73 15 0
Script.Message: 74 14 0
Script.Message: 75 14 1
Script.Message: 76 14 2
Script.Message: 77 15 2
Script.Message: 78 15 1
Script.Message: 79 16 1
Script.Message: 80 15 1
Script.Message: 81 14 1
Script.Message: 82 14 2
Script.Message: 83 14 1
Script.Message: 84 14 0
Script.Message: 85 15 0
Script.Message: 86 16 0
Script.Message: 87 17 0
Script.Message: 88 18 0
Script.Message: 89 17 0
Script.Message: 90 18 0
Script.Message: 91 19 0
Script.Message: 92 20 0
Script.Message: 93 19 0
Script.Message: 94 20 0
Script.Message: 95 21 0
Script.Message: 96 21 1
Script.Message: 97 21 0
Script.Message: 98 20 0
Script.Message: 99 19 0
Script.Message: 100 18 0
Script.Message: 101 17 0
Script.Message: 102 17 1
Script.Message: 103 17 0
Script.Message: 104 16 0
Script.Message: 105 17 0
Script.Message: 106 17 1
Script.Message: 107 18 1
Script.Message: 108 17 1
Script.Message: 109 17 0
Script.Message: 110 17 1
Script.Message: 111 17 2
Script.Message: 112 17 1
Script.Message: 113 16 1
Script.Message: 114 17 1
Script.Message: 115 16 1
Script.Message: 116 17 1
Script.Message: 117 16 1
Script.Message: 118 16 0
Script.Message: 119 17 0
Script.Message: 120 16 0
Script.Message: 121 15 0
Script.Message: 122 14 0
Script.Message: 123 15 0
Script.Message: 124 15 1
Script.Message: 125 15 0
Script.Message: 126 16 0
Script.Message: 127 17 0
Script.Message: 128 17 1
Script.Message: 129 17 2
Script.Message: 130 17 3
Script.Message: 131 16 3
Script.Message: 132 16 2
Script.Message: 133 17 2
Script.Message: 134 17 3
Script.Message: 135 16 3
Script.Message: 136 16 4
Script.Message: 137 16 5
Script.Message: 138 16 4
Script.Message: 139 16 3
Script.Message: 140 17 3
Script.Message: 141 17 2
Script.Message: 142 16 2
Script.Message: 143 17 2
Script.Message: 144 17 1
Script.Message: 145 17 0
Script.Message: 146 18 0
Script.Message: 147 19 0
Script.Message: 148 18 0
Script.Message: 149 18 1
Script.Message: 150 19 1
Script.Message: 151 19 0
Script.Message: 152 18 0
Script.Message: 153 17 0
Script.Message: 154 17 1
Script.Message: 155 17 2
Script.Message: 156 18 2
Script.Message: 157 18 1
Script.Message: 158 17 0
Script.Message: 159 18 0
Script.Message: 160 19 0
Script.Message: 161 20 0
Script.Message: 162 19 0
Script.Message: 163 18 0
Script.Message: 164 18 1
Script.Message: 165 18 2
Script.Message: 166 17 2
Script.Message: 167 16 2
Script.Message: 168 16 1
Script.Message: 169 15 1
Script.Message: 170 14 1
Script.Message: 171 14 2
Script.Message: 172 13 2
Script.Message: 173 13 3
Script.Message: 174 13 4
Script.Message: 175 13 3
Script.Message: 176 13 2
Script.Message: 177 13 1
Script.Message: 178 13 2
Script.Message: 179 14 2
Script.Message: 180 13 2
Script.Message: 181 14 2
Script.Message: 182 14 3
Script.Message: 183 13 2
Script.Message: 184 13 3
Script.Message: 185 13 4
Script.Message: 186 13 5
Script.Message: 187 13 6
Script.Message: 188 13 7
Script.Message: 189 13 8
Script.Message: 190 13 7
Script.Message: 191 13 8
Script.Message: 192 14 8
Script.Message: 193 13 8
Script.Message: 194 13 7
Script.Message: 195 13 6
Script.Message: 196 12 5
Script.Message: 197 13 5
Script.Message: 198 13 6
Script.Message: 199 14 6
    """,
    """Script.Message: 0 9 19
Script.Message: 1 10 19
Script.Message: 2 10 18
Script.Message: 3 10 17
Script.Message: 4 10 16
Script.Message: 5 9 11
Script.Message: 6 10 11
Script.Message: 7 11 11
Script.Message: 8 12 11
Script.Message: 9 11 7
Script.Message: 10 11 6
Script.Message: 11 11 7
Script.Message: 12 10 6
Script.Message: 13 11 6
Script.Message: 14 12 6
Script.Message: 15 13 6
Script.Message: 16 14 6
Script.Message: 17 14 5
Script.Message: 18 15 5
Script.Message: 19 15 4
Script.Message: 20 16 4
Script.Message: 21 16 3
Script.Message: 22 16 4
Script.Message: 23 15 3
Script.Message: 24 14 3
Script.Message: 25 15 3
Script.Message: 26 15 2
Script.Message: 27 15 1
Script.Message: 28 14 1
Script.Message: 29 15 1
Script.Message: 30 15 2
Script.Message: 31 15 1
Script.Message: 32 16 1
Script.Message: 33 15 1
Script.Message: 34 14 1
Script.Message: 35 13 1
Script.Message: 36 13 2
Script.Message: 37 14 2
Script.Message: 38 14 1
Script.Message: 39 15 1
Script.Message: 40 14 1
Script.Message: 41 13 1
Script.Message: 42 14 1
Script.Message: 43 13 1
Script.Message: 44 13 0
Script.Message: 45 13 1
Script.Message: 46 13 0
Script.Message: 47 14 0
Script.Message: 48 15 0
Script.Message: 49 16 0
Script.Message: 50 17 0
Script.Message: 51 17 1
Script.Message: 52 17 2
Script.Message: 53 18 2
Script.Message: 54 17 2
Script.Message: 55 18 2
Script.Message: 56 18 1
Script.Message: 57 18 2
Script.Message: 58 19 2
Script.Message: 59 18 2
Script.Message: 60 18 3
Script.Message: 61 18 2
Script.Message: 62 17 2
Script.Message: 63 17 1
Script.Message: 64 17 0
Script.Message: 65 18 0
Script.Message: 66 19 0
Script.Message: 67 20 0
Script.Message: 68 20 1
Script.Message: 69 19 1
Script.Message: 70 20 1
Script.Message: 71 20 2
Script.Message: 72 21 2
Script.Message: 73 21 1
Script.Message: 74 21 0
Script.Message: 75 21 1
Script.Message: 76 20 1
Script.Message: 77 20 0
Script.Message: 78 20 1
Script.Message: 79 20 2
Script.Message: 80 19 2
Script.Message: 81 20 2
Script.Message: 82 21 2
Script.Message: 83 22 2
Script.Message: 84 23 2
Script.Message: 85 23 1
Script.Message: 86 24 1
Script.Message: 87 24 2
Script.Message: 88 24 3
Script.Message: 89 25 3
Script.Message: 90 26 3
Script.Message: 91 25 3
Script.Message: 92 26 3
Script.Message: 93 26 4
Script.Message: 94 26 5
Script.Message: 95 26 4
Script.Message: 96 26 5
Script.Message: 97 26 4
Script.Message: 98 26 5
Script.Message: 99 27 5
Script.Message: 100 28 5
Script.Message: 101 28 4
Script.Message: 102 28 3
Script.Message: 103 29 3
Script.Message: 104 30 3
Script.Message: 105 29 3
Script.Message: 106 29 2
Script.Message: 107 28 2
Script.Message: 108 28 1
Script.Message: 109 28 0
Script.Message: 110 27 0
Script.Message: 111 26 0
Script.Message: 112 27 0
Script.Message: 113 27 1
Script.Message: 114 27 0
Script.Message: 115 27 1
Script.Message: 116 27 2
Script.Message: 117 27 3
Script.Message: 118 27 4
Script.Message: 119 28 4
Script.Message: 120 28 5
Script.Message: 121 28 4
Script.Message: 122 28 3
Script.Message: 123 28 4
Script.Message: 124 29 4
Script.Message: 125 28 3
Script.Message: 126 28 4
Script.Message: 127 28 3
Script.Message: 128 28 4
Script.Message: 129 28 5
Script.Message: 130 29 5
Script.Message: 131 30 5
Script.Message: 132 30 6
Script.Message: 133 31 6
Script.Message: 134 31 5
Script.Message: 135 31 6
Script.Message: 136 32 6
Script.Message: 137 31 5
Script.Message: 138 32 5
Script.Message: 139 31 4
Script.Message: 140 30 4
Script.Message: 141 30 5
Script.Message: 142 29 4
Script.Message: 143 29 3
Script.Message: 144 29 4
Script.Message: 145 29 3
Script.Message: 146 29 4
Script.Message: 147 29 3
Script.Message: 148 28 3
Script.Message: 149 29 3
Script.Message: 150 29 4
Script.Message: 151 28 4
Script.Message: 152 28 5
Script.Message: 153 28 4
Script.Message: 154 29 4
Script.Message: 155 30 4
Script.Message: 156 30 5
Script.Message: 157 30 6
Script.Message: 158 31 6
Script.Message: 159 31 7
Script.Message: 160 31 8
Script.Message: 161 30 6
Script.Message: 162 29 6
Script.Message: 163 29 7
Script.Message: 164 30 7
Script.Message: 165 30 8
Script.Message: 166 30 9
Script.Message: 167 29 9
Script.Message: 168 28 8
Script.Message: 169 29 8
Script.Message: 170 29 9
Script.Message: 171 29 10
Script.Message: 172 30 10
Script.Message: 173 30 11
Script.Message: 174 30 12
Script.Message: 175 30 13
Script.Message: 176 30 12
Script.Message: 177 31 12
Script.Message: 178 31 11
Script.Message: 179 31 10
Script.Message: 180 30 10
Script.Message: 181 30 11
Script.Message: 182 30 12
Script.Message: 183 31 12
Script.Message: 184 32 12
Script.Message: 185 32 13
Script.Message: 186 31 11
Script.Message: 187 31 12
Script.Message: 188 31 11
Script.Message: 189 32 11
Script.Message: 190 32 10
Script.Message: 191 33 10
Script.Message: 192 32 8
Script.Message: 193 32 7
Script.Message: 194 32 8
Script.Message: 195 33 8
Script.Message: 196 32 6
Script.Message: 197 32 7
Script.Message: 198 32 8
Script.Message: 199 31 8
    """
]

def chart(data: [[[int]]], index: int, name):
    fig = plt.figure(figsize=(8, 8))
    for i, d in enumerate(data):
        serie = [s[index] for s in d]
        plt.plot(serie, label=str(i))
    ax = fig.gca()
    ax.set_xlabel('Numer ewolucji')
    ax.set_ylabel('Liczba ' + ("połączeń" if index == 1 else "neuronów"))
    ax.legend([1,2,3,4,5])
    plt.show()
    write_pdf(fig, name)


if __name__ == '__main__':
    chunked = []
    for c in charts:
        chunks = [chunk for chunk in c.split("\n") if len(chunk.strip()) > 0]
        values = [[int(v) for v in chunk.split()[2:4]] for chunk in chunks]
        chunked.append(values)

    chart(chunked, 0, "neurons-gen")
    chart(chunked, 1, "connections-gen")
