import pathlib
import re
import subprocess
import unittest
import tempfile
import os

class TestLab2(unittest.TestCase):

    def _make():
        result = subprocess.run(['make'], capture_output=True, text=True)
        return result

    def _make_clean():
        result = subprocess.run(['make', 'clean'],
                                capture_output=True, text=True)
        return result

    @classmethod
    def setUpClass(cls):
        cls.make = cls._make().returncode == 0

    @classmethod
    def tearDownClass(cls):
        cls._make_clean()

    def test_averages(self):

        fileName="processes.txt"
        correctAvgWaitTime=(0,5.5, 5.0,  7,   4.5,  5.5,  6.25, 4.75)
        correctAvgRespTime=(0,0.75,1.5,2.75,  3.25, 3.25, 4,    4.75)

        self.assertTrue(self.make, msg='make failed')
        for x in range(1,7):

            cl_result = subprocess.check_output(('./rr',fileName,str(x))).decode()
            lines=cl_result.split('\n')
            testAvgWaitTime=float(lines[0].split(':')[1])
            testAvgRespTime=float(lines[1].split(':')[1])

            result=True
            if(testAvgWaitTime!=correctAvgWaitTime[x]):
                result=False
            if(testAvgRespTime!=correctAvgRespTime[x]):
                result=False
            
            self.assertTrue(result,f"\n    Quantum Time: {x}\n Correct Results: Avg Wait. Time:{correctAvgWaitTime[x]}, Avg. Resp. Time:{correctAvgRespTime[x]}\n    Your Results: Avg Wait. Time:{testAvgWaitTime}, Avg. Resp. Time:{testAvgRespTime}\n")

    def test_arrival_and_requeue(self):
            self.assertTrue(self.make, msg='make failed')

            correctAvgWaitTime=(0, 5,    5.25,  6.5,   4.0,  4.5,  5.75,   4.75)
            correctAvgRespTime=(0, 0.75, 1.5,   2.25,  2.75, 3.25, 3.5,    4.75)

            #temp file for this case.
            with tempfile.NamedTemporaryFile() as f:

                f.write(b'4\n')
                f.write(b'1, 0, 7\n')
                f.write(b'2, 3, 4\n')
                f.write(b'3, 4, 1\n')
                f.write(b'4, 6, 4\n')
                f.flush()

                for x in range(1,7):
                    cl_result = subprocess.check_output(('./rr',f.name,str(x))).decode()
                    lines=cl_result.split('\n')
                    testAvgWaitTime=float(lines[0].split(':')[1])
                    testAvgRespTime=float(lines[1].split(':')[1])

                    result=True
                    if(testAvgWaitTime!=correctAvgWaitTime[x]):
                        result=False
                    if(testAvgRespTime!=correctAvgRespTime[x]):
                        result=False

                    self.assertTrue(result,f"\n Cannot handle re-queue and new process arrival at the same time\n   Quantum Time: {x}\n Correct Results: Avg Wait. Time:{correctAvgWaitTime[x]}, Avg. Resp. Time:{correctAvgRespTime[x]}\n    Your Results: Avg Wait. Time:{testAvgWaitTime}, Avg. Resp. Time:{testAvgRespTime}\n")

