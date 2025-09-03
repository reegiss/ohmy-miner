// include/miner/Plugin.h

/*
 * Copyright (C) 2025 Regis Araujo Melo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

// Opaque pointer to the C++ interface to hide implementation details.
struct IAlgorithm_t;

#ifdef __cplusplus
extern "C" {
#endif

// C-style API for creating and destroying an algorithm instance.
// These functions must be exported by each algorithm plugin (.so file).

// Creates an instance of the algorithm.
// Returns a pointer to the opaque struct, or NULL on failure.
struct IAlgorithm_t* create_algorithm();

// Destroys an instance of the algorithm.
void destroy_algorithm(struct IAlgorithm_t* algo);

#ifdef __cplusplus
}
#endif