<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================
[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tlcpack.ai/buildStatus/icon?job=tvm/main)](https://ci.tlcpack.ai/job/tvm/job/main/)
[![WinMacBuild](https://github.com/apache/tvm/workflows/WinMacBuild/badge.svg)](https://github.com/apache/tvm/actions?query=workflow%3AWinMacBuild)

Apache TVM is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

License
-------
TVM is licensed under the [Apache-2.0](LICENSE) license.


# Build, Execution Note

profiling.cc는 전처리 옵션에 따라 빌드 방식이 달라짐.

## 1. 튜닝할 때

NVML 기반 파워 측정을 끄고 빌드함.

cmake .. -DTVM_ENABLE_NVML_POWER=OFF
make -j$(nproc)

이후 튜닝 스크립트 실행.
python e2e_3.py

## 2. 파워 측정할 때

NVML 기반 파워 측정을 켜고 빌드함.

cmake .. -DTVM_ENABLE_NVML_POWER=ON
make -j$(nproc)

이후 파워 데이터셋 생성 실행.
python append_dataset.py



# Warmup logic
tutorials에 warmup log 생성됨. 

warmup log의 final power랑 terminal에 뜨는 건 본측정값과 비교해서 같아질 떄까지 sliding window 사이즈 조정함.

시간 관계상 커널 크기에 따라 sliding window로직 달라지도록 조정함.


