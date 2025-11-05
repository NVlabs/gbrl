//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/gbrl/license.html
//
//////////////////////////////////////////////////////////////////////////////

/**
 * @file config.h
 * @brief Version configuration for the GBRL library
 * 
 * This header defines the semantic versioning constants for the GBRL library.
 * The version follows the pattern MAJOR.MINOR.PATCH where:
 * - MAJOR version indicates incompatible API changes
 * - MINOR version indicates added functionality in a backward compatible manner
 * - PATCH version indicates backward compatible bug fixes
 */

#ifndef VERSION_CONFIG_H
#define VERSION_CONFIG_H

/** @brief Major version number - incremented for incompatible API changes */
#define MAJOR_VERSION 1

/** @brief Minor version number - incremented for backward compatible new features */
#define MINOR_VERSION 1

/** @brief Patch version number - incremented for backward compatible bug fixes */
#define PATCH_VERSION 2

#endif // VERSION_CONFIG_H