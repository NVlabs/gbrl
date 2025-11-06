//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024-2025, NVIDIA Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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
#define PATCH_VERSION 4

#endif // VERSION_CONFIG_H