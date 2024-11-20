#!/bin/bash
WHEEL_FILENAME="gbrl-1.0.4-cp311-cp311-linux_x86_64.whl"

# License expected to be found in your package:
EXPECTED_PKG_LICENSE="NVIDIA Proprietary Software"

# Comma separated list of rules that are ignored (e.g "B301,B303,B304")
# See: https://bandit.readthedocs.io/en/latest/blacklists/blacklist_calls.html
# **MUST READ:** https://gitlab-master.nvidia.com/dl/pypi/Wheel-CI-CD/-/tree/master#important-notice-do-not-skip-over-this-section
# Example: SKIPPED_SECURITY_RULES="B303,B306"
SKIPPED_SECURITY_RULES=""

# Pre-approved number of times where `# nosec` is used to skip security check.
# This method shall only be used on false alarms. Any pre-approved potentially
# risky code needs to go through the variable `SKIPPED_SECURITY_RULES`.
# **MUST READ:** https://gitlab-master.nvidia.com/dl/pypi/Wheel-CI-CD/-/tree/master#important-notice-do-not-skip-over-this-section
ALLOWED_NOSEC_COUNT="0"

cd dist
sudo docker pull gitlab-master.nvidia.com:5005/dl/pypi/wheel-ci-cd:wheeltamer
sudo docker run --rm --network=host \
	    -e EXPECTED_PKG_LICENSE="${EXPECTED_PKG_LICENSE}" \
	        -e SKIPPED_SECURITY_RULES="${SKIPPED_SECURITY_RULES}" \
		    -e ALLOWED_NOSEC_COUNT="${ALLOWED_NOSEC_COUNT}" \
		        -v $(pwd)/${WHEEL_FILENAME}:/workspace/${WHEEL_FILENAME} \
			    gitlab-master.nvidia.com:5005/dl/pypi/wheel-ci-cd:wheeltamer

# `component_name`
# Required
# The reference name of the project to be published.
# IMPORTANT: This usually matches the wheel name, but may diverge in cases where multiple wheels are to be released using a single configuration.
# `os`
# Required
# The target os platform for the wheel. Usually "linux", "windows", "any".
# `arch`
# Required
# The supported cpu architecture of the wheel. Typically "x86_64" or "aarch64".
# If a wheel supports all architectures, then the arch property should be set to "any".
# `version`
# Required
# The version. Must obey python standard versioning. See PEP-440
# `branch`
# Required 
# The branch. Typically used to identify cuda toolkit version the artifact targets.
# `release_approver`
# Required
# The individual responsible for setting the wheel as ready to ship.
# `release_status`
