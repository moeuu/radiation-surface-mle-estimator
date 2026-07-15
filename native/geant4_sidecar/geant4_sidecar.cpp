#include <G4Box.hh>
#include <G4EmCalculator.hh>
#include <G4EmStandardPhysics_option4.hh>
#include <G4Event.hh>
#include <G4Gamma.hh>
#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#ifdef G4MULTITHREADED
#include <G4MTRunManager.hh>
#endif
#include <G4NistManager.hh>
#include <G4PVPlacement.hh>
#include <G4ParticleGun.hh>
#include <G4PhysListFactory.hh>
#include <G4PrimaryParticle.hh>
#include <G4PrimaryVertex.hh>
#include <G4RotationMatrix.hh>
#include <G4RunManager.hh>
#include <G4RunManagerFactory.hh>
#include <G4SDManager.hh>
#include <G4Sphere.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4TessellatedSolid.hh>
#include <G4ThreeVector.hh>
#include <G4TriangularFacet.hh>
#include <G4Track.hh>
#include <G4Types.hh>
#include <G4UserEventAction.hh>
#include <G4UserStackingAction.hh>
#include <G4UserSteppingAction.hh>
#include <G4VModularPhysicsList.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VProcess.hh>
#include <G4VSensitiveDetector.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4VisAttributes.hh>
#include <G4ios.hh>
#include <Randomize.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kDefaultCrystalRadiusM = 0.038;
constexpr double kDefaultCrystalLengthM = 0.076;
constexpr double kDefaultHousingThicknessM = 0.0015;
constexpr double kDefaultShieldContactRadiusM = kDefaultCrystalRadiusM + kDefaultHousingThicknessM;
constexpr double kDefaultShieldTransmissionScale = 0.6989700043360189;
constexpr double kDefaultFeShieldTvlThicknessM = 0.05;
constexpr double kDefaultPbShieldTvlThicknessM = 0.022;
constexpr double kDefaultFeShieldThicknessM =
    kDefaultFeShieldTvlThicknessM * kDefaultShieldTransmissionScale;
constexpr double kDefaultPbShieldThicknessM =
    kDefaultPbShieldTvlThicknessM * kDefaultShieldTransmissionScale;
constexpr double kDefaultFeShieldInnerRadiusM = kDefaultShieldContactRadiusM;
constexpr double kDefaultPbShieldInnerRadiusM =
    kDefaultFeShieldInnerRadiusM + kDefaultFeShieldThicknessM;

struct MaterialSpec {
    std::string name;
    double density_g_cm3 = -1.0;
    std::string preset_name;
    std::map<std::string, double> composition_by_mass;
};

struct VolumeSpec {
    std::string path;
    std::string shape;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double sx = -1.0;
    double sy = -1.0;
    double sz = -1.0;
    double radius_m = -1.0;
    MaterialSpec material;
    std::vector<std::array<double, 9>> triangles;
    std::string transport_group;
    std::string transport_mode = "geant4";
};

struct SourceSpec {
    std::string isotope;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double intensity_cps_1m = 0.0;
};

struct DetectorSpec {
    double crystal_radius_m = kDefaultCrystalRadiusM;
    double crystal_length_m = kDefaultCrystalLengthM;
    double housing_thickness_m = kDefaultHousingThicknessM;
    std::string crystal_shape = "sphere";
    std::string crystal_material = "cebr3";
    std::string housing_material = "aluminum";
};

struct ShieldSpec {
    std::string kind;
    std::string path;
    std::string shape = "spherical_octant_shell";
    double inner_radius_m = kDefaultFeShieldInnerRadiusM;
    double outer_radius_m = kDefaultFeShieldInnerRadiusM + kDefaultFeShieldThicknessM;
    double thickness_m = kDefaultFeShieldThicknessM;
    double sx = 0.25;
    double sy = 0.08;
    double sz = 0.25;
    MaterialSpec material;
};

ShieldSpec DefaultFeShieldSpec() {
    ShieldSpec shield;
    shield.kind = "fe";
    shield.shape = "spherical_octant_shell";
    shield.inner_radius_m = kDefaultFeShieldInnerRadiusM;
    shield.thickness_m = kDefaultFeShieldThicknessM;
    shield.outer_radius_m = shield.inner_radius_m + shield.thickness_m;
    shield.material.name = "fe";
    shield.material.preset_name = "iron";
    return shield;
}

ShieldSpec DefaultPbShieldSpec() {
    ShieldSpec shield;
    shield.kind = "pb";
    shield.shape = "spherical_octant_shell";
    shield.inner_radius_m = kDefaultPbShieldInnerRadiusM;
    shield.thickness_m = kDefaultPbShieldThicknessM;
    shield.outer_radius_m = shield.inner_radius_m + shield.thickness_m;
    shield.material.name = "pb";
    shield.material.preset_name = "lead";
    return shield;
}

struct PoseSpec {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
};

struct SceneSpec {
    std::string scene_hash;
    std::string usd_path;
    double room_x = 10.0;
    double room_y = 20.0;
    double room_z = 10.0;
    DetectorSpec detector;
    ShieldSpec fe_shield = DefaultFeShieldSpec();
    ShieldSpec pb_shield = DefaultPbShieldSpec();
    std::vector<SourceSpec> sources;
    std::vector<VolumeSpec> volumes;
};

struct RequestSpec {
    int step_id = 0;
    double dwell_time_s = 1.0;
    long seed = 123;
    PoseSpec detector_pose;
    PoseSpec fe_pose;
    PoseSpec pb_pose;
};

struct LineSpec {
    double energy_keV;
    double intensity;
};

struct SimulationResult {
    std::vector<double> spectrum_counts;
    std::vector<double> spectrum_count_variance;
    std::map<std::string, std::string> metadata;
};

struct TransportOptions {
    double background_cps = 0.0;
    std::string source_rate_model = "detector_cps_1m";
    std::string source_bias_mode = "detector_cone";
    double source_bias_cone_half_angle_deg = 0.0;
    double source_bias_isotropic_fraction = 0.1;
    std::string detector_scoring_mode = "full_transport";
    std::string secondary_transport_mode = "full_transport";
    double primary_sampling_fraction = 1.0;
};

struct EnergyDeposit {
    double energy_keV = 0.0;
    double weight = 1.0;
};

enum class DetectorEntryClass {
    kUncollidedPrimary = 0,
    kInteractedPrimary = 1,
    kSecondary = 2,
};

struct WeightedEventDeposit {
    double edep_mev = 0.0;
    double weight = 1.0;
    DetectorEntryClass entry_class = DetectorEntryClass::kUncollidedPrimary;
};

DetectorEntryClass MergeDetectorEntryClass(
    const DetectorEntryClass current,
    const DetectorEntryClass incoming
) {
    return static_cast<int>(incoming) > static_cast<int>(current) ? incoming : current;
}

std::string NormalizeToken(const std::string& token) {
    std::string result = token;
    std::size_t pos = 0;
    while ((pos = result.find("%20", pos)) != std::string::npos) {
        result.replace(pos, 3, " ");
        pos += 1;
    }
    return result;
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string NormalizeModeToken(std::string value) {
    value = ToLower(std::move(value));
    for (char& ch : value) {
        if (ch == '-') {
            ch = '_';
        }
    }
    return value;
}

std::string JoinSet(const std::set<std::string>& values, const std::string& separator) {
    std::ostringstream stream;
    bool first = true;
    for (const auto& value : values) {
        if (!first) {
            stream << separator;
        }
        stream << value;
        first = false;
    }
    return stream.str();
}

std::string SanitizeMetadataToken(std::string value) {
    for (char& ch : value) {
        if (std::isspace(static_cast<unsigned char>(ch)) || ch == ',' || ch == '=') {
            ch = '_';
        }
    }
    return value.empty() ? "unknown" : value;
}

std::string SerializeCounterMap(const std::map<std::string, long long>& values) {
    if (values.empty()) {
        return "-";
    }
    std::ostringstream stream;
    bool first = true;
    for (const auto& item : values) {
        if (!first) {
            stream << ",";
        }
        first = false;
        stream << SanitizeMetadataToken(item.first) << ":" << item.second;
    }
    return stream.str();
}

std::string SerializeTopCounterMap(
    const std::map<std::string, long long>& values,
    const std::size_t limit
) {
    std::vector<std::pair<std::string, long long>> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) {
            return lhs.second > rhs.second;
        }
        return lhs.first < rhs.first;
    });
    if (sorted.size() > limit) {
        sorted.resize(limit);
    }
    std::map<std::string, long long> top(sorted.begin(), sorted.end());
    return SerializeCounterMap(top);
}

std::map<std::string, std::string> ParseFields(const std::vector<std::string>& tokens, std::size_t start_index = 1) {
    std::map<std::string, std::string> fields;
    for (std::size_t index = start_index; index < tokens.size(); ++index) {
        const auto separator = tokens[index].find('=');
        if (separator == std::string::npos) {
            continue;
        }
        fields[tokens[index].substr(0, separator)] = NormalizeToken(tokens[index].substr(separator + 1));
    }
    return fields;
}

std::vector<std::string> Split(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

double ParseDouble(const std::map<std::string, std::string>& fields, const std::string& key, double fallback = 0.0) {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second == "-") {
        return fallback;
    }
    return std::stod(it->second);
}

long ParseLong(const std::map<std::string, std::string>& fields, const std::string& key, long fallback = 0) {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second == "-") {
        return fallback;
    }
    return std::stol(it->second);
}

std::string ParseString(const std::map<std::string, std::string>& fields, const std::string& key, const std::string& fallback = "") {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second == "-") {
        return fallback;
    }
    return it->second;
}

std::vector<LineSpec> GammaLinesForIsotope(const std::string& isotope) {
    if (isotope == "Cs-137") {
        return {{662.0, 0.85}};
    }
    if (isotope == "Co-60") {
        return {{1173.0, 0.5}, {1332.0, 0.5}};
    }
    if (isotope == "Eu-154") {
        return {
            {723.3, 0.25},
            {873.2, 0.14},
            {996.3, 0.14},
            {1274.5, 0.45},
            {1494.0, 0.01},
            {1596.5, 0.02},
        };
    }
    return {};
}

double SigmaEnergyKeV(const double energy_keV) {
    return std::max(0.5 * std::sqrt(std::max(0.0, energy_keV)) - 1.5, 0.5);
}

double BackgroundShape(const double energy_keV) {
    const double low_energy_scatter = 0.62 * std::exp(-std::max(0.0, energy_keV) / 260.0);
    const double long_tail = 0.30 * std::exp(-std::max(0.0, energy_keV) / 1050.0);
    const double potassium_line = 0.08 * std::exp(
        -0.5 * std::pow((energy_keV - 1460.0) / 38.0, 2.0)
    );
    return std::max(0.0, low_energy_scatter + long_tail + potassium_line);
}

void AddBackgroundSpectrum(
    std::vector<double>& spectrum,
    std::vector<double>* spectrum_variance,
    const double bin_width_keV,
    const double dwell_time_s,
    const TransportOptions& options,
    std::mt19937_64& rng
) {
    if (options.background_cps <= 0.0 || spectrum.empty()) {
        return;
    }
    std::vector<double> shape(spectrum.size(), 0.0);
    double normalization = 0.0;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        const double energy_keV = (static_cast<double>(index) + 0.5) * bin_width_keV;
        shape[index] = BackgroundShape(energy_keV);
        normalization += shape[index];
    }
    if (normalization <= 0.0) {
        return;
    }
    const double expected_total = options.background_cps * std::max(0.0, dwell_time_s);
    for (std::size_t index = 0; index < spectrum.size(); ++index) {
        const double expected = expected_total * shape[index] / normalization;
        std::poisson_distribution<long> distribution(std::max(0.0, expected));
        const double sampled = static_cast<double>(distribution(rng));
        spectrum[index] += sampled;
        if (spectrum_variance != nullptr && index < spectrum_variance->size()) {
            (*spectrum_variance)[index] += expected;
        }
    }
}

double InverseSquareScale(
    const double source_x,
    const double source_y,
    const double source_z,
    const double detector_x,
    const double detector_y,
    const double detector_z
) {
    const double dx = detector_x - source_x;
    const double dy = detector_y - source_y;
    const double dz = detector_z - source_z;
    const double distance_sq = dx * dx + dy * dy + dz * dz;
    if (distance_sq <= 1.0e-12) {
        return 0.0;
    }
    return 1.0 / distance_sq;
}

double SphereSolidAngleFraction(const double distance_m, const double radius_m) {
    const double radius = std::max(0.0, radius_m);
    if (radius <= 0.0) {
        return 0.0;
    }
    if (distance_m <= radius) {
        return 0.5;
    }
    const double ratio = std::clamp(radius / std::max(distance_m, 1.0e-12), 0.0, 1.0);
    return 0.5 * (1.0 - std::sqrt(std::max(0.0, 1.0 - ratio * ratio)));
}

double DetectorCpsGeometryScale(
    const double source_x,
    const double source_y,
    const double source_z,
    const double detector_x,
    const double detector_y,
    const double detector_z,
    const DetectorSpec& detector
) {
    const double dx = detector_x - source_x;
    const double dy = detector_y - source_y;
    const double dz = detector_z - source_z;
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (distance <= 1.0e-12) {
        return 0.0;
    }
    const double radius = std::max(0.0, detector.crystal_radius_m);
    if (radius <= 0.0) {
        return InverseSquareScale(
            source_x,
            source_y,
            source_z,
            detector_x,
            detector_y,
            detector_z
        );
    }
    const double reference_distance = std::max(1.0, radius);
    const double reference_fraction = std::max(
        SphereSolidAngleFraction(reference_distance, radius),
        1.0e-12
    );
    return SphereSolidAngleFraction(std::max(distance, radius), radius)
        / reference_fraction;
}

double DetectorReferenceAcceptance(const DetectorSpec& detector) {
    constexpr double kReferenceDistanceM = 1.0;
    constexpr double kPi = 3.14159265358979323846;
    const double radius_m = std::max(1.0e-9, detector.crystal_radius_m);
    const double face_area_m2 = kPi * radius_m * radius_m;
    const double sphere_area_m2 = 4.0 * kPi * kReferenceDistanceM * kReferenceDistanceM;
    return std::clamp(face_area_m2 / sphere_area_m2, 1.0e-12, 1.0);
}

std::string NormalizeSourceBiasMode(const std::string& mode) {
    auto normalized = NormalizeModeToken(mode);
    if (normalized.empty() || normalized == "none" || normalized == "isotropic") {
        return "analog";
    }
    if (normalized == "mixture_cone" || normalized == "cone_isotropic") {
        return "mixture_cone_isotropic";
    }
    if (normalized == "detector" || normalized == "detector_directed" || normalized == "detector_cone") {
        return "detector_cone";
    }
    return normalized;
}

bool UsesSourceBias(const TransportOptions& options) {
    return NormalizeSourceBiasMode(options.source_bias_mode) == "mixture_cone_isotropic";
}

std::string NormalizeSourceRateModel(const std::string& mode) {
    auto normalized = NormalizeModeToken(mode);
    if (
        normalized.empty()
        || normalized == "detector"
        || normalized == "detector_cps"
        || normalized == "detector_cps_1m"
        || normalized == "detector_count_rate"
    ) {
        return "detector_cps_1m";
    }
    if (
        normalized == "isotropic"
        || normalized == "isotropic_emission"
        || normalized == "isotropic_emission_equivalent"
        || normalized == "emission_equivalent"
    ) {
        return "isotropic_emission_equivalent";
    }
    return normalized;
}

std::string NormalizeDetectorScoringMode(const std::string& mode) {
    const auto normalized = NormalizeModeToken(mode);
    if (
        normalized.empty()
        || normalized == "full"
        || normalized == "full_transport"
        || normalized == "energy_deposit"
    ) {
        return "full_transport";
    }
    if (
        normalized == "fast"
        || normalized == "incident_energy"
        || normalized == "incident_gamma_energy"
        || normalized == "perfect_absorption"
    ) {
        return "incident_gamma_energy";
    }
    return normalized;
}

bool UsesFastDetectorScoring(const TransportOptions& options) {
    return NormalizeDetectorScoringMode(options.detector_scoring_mode) == "incident_gamma_energy";
}

std::string NormalizeSecondaryTransportMode(const std::string& mode) {
    const auto normalized = NormalizeModeToken(mode);
    if (
        normalized.empty()
        || normalized == "full"
        || normalized == "full_transport"
        || normalized == "all_particles"
    ) {
        return "full_transport";
    }
    if (
        normalized == "gamma_only"
        || normalized == "photon_only"
        || normalized == "kill_charged"
        || normalized == "kill_charged_secondaries"
    ) {
        return "gamma_only";
    }
    return normalized;
}

bool UsesGammaOnlySecondaryTransport(const TransportOptions& options) {
    return NormalizeSecondaryTransportMode(options.secondary_transport_mode) == "gamma_only";
}

double DetectorTargetRadiusM(const DetectorSpec& detector) {
    return std::max(
        1.0e-9,
        detector.crystal_radius_m + std::max(0.0, detector.housing_thickness_m)
    );
}

double EffectiveConeHalfAngleRad(
    const SourceSpec& source,
    const SceneSpec& scene,
    const RequestSpec& request,
    const TransportOptions& options
) {
    const double dx = request.detector_pose.x - source.x;
    const double dy = request.detector_pose.y - source.y;
    const double dz = request.detector_pose.z - source.z;
    const double distance_m = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double target_radius_m = DetectorTargetRadiusM(scene.detector);
    double covering_angle = CLHEP::pi;
    if (distance_m > target_radius_m) {
        covering_angle = std::asin(std::clamp(target_radius_m / distance_m, 0.0, 1.0));
    }
    const double configured_angle = std::max(0.0, options.source_bias_cone_half_angle_deg) * CLHEP::pi / 180.0;
    return std::clamp(std::max(covering_angle, configured_angle), 1.0e-9, CLHEP::pi);
}

double ConeSolidAngleSr(const double half_angle_rad) {
    const double theta = std::clamp(half_angle_rad, 0.0, CLHEP::pi);
    return std::max(2.0 * CLHEP::pi * (1.0 - std::cos(theta)), 1.0e-18);
}

bool UseTheoryTvlProfile(const std::string& physics_profile) {
    std::string profile = physics_profile;
    std::transform(profile.begin(), profile.end(), profile.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return profile.find("theory_tvl") != std::string::npos || profile.find("ideal_tvl") != std::string::npos;
}

double MuFromTvlMm(const double tvl_mm) {
    return std::log(10.0) / (std::max(tvl_mm, 1.0e-12) / 10.0);
}

double TvlMmForShield(const std::string& isotope, const std::string& shield_kind) {
    const bool is_fe = shield_kind == "fe";
    if (isotope == "Cs-137") {
        return is_fe ? 50.0 : 22.0;
    }
    if (isotope == "Co-60") {
        return is_fe ? 67.0 : 40.0;
    }
    if (isotope == "Eu-154") {
        return is_fe ? 57.7 : 28.1;
    }
    return is_fe ? 50.0 : 22.0;
}

std::array<double, 3> ShieldNormalFromPose(const PoseSpec& pose) {
    const double norm = std::sqrt(
        pose.qw * pose.qw + pose.qx * pose.qx + pose.qy * pose.qy + pose.qz * pose.qz
    );
    if (norm <= 1.0e-12) {
        const double inv_sqrt3 = 1.0 / std::sqrt(3.0);
        return {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    }
    const double w = pose.qw / norm;
    const double x = pose.qx / norm;
    const double y = pose.qy / norm;
    const double z = pose.qz / norm;
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - z * w);
    const double r02 = 2.0 * (x * z + y * w);
    const double r10 = 2.0 * (x * y + z * w);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - x * w);
    const double r20 = 2.0 * (x * z - y * w);
    const double r21 = 2.0 * (y * z + x * w);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);
    const double local = 1.0 / std::sqrt(3.0);
    std::array<double, 3> normal = {
        r00 * local + r01 * local + r02 * local,
        r10 * local + r11 * local + r12 * local,
        r20 * local + r21 * local + r22 * local,
    };
    const double mag = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (mag <= 1.0e-12) {
        return {local, local, local};
    }
    return {normal[0] / mag, normal[1] / mag, normal[2] / mag};
}

std::array<std::array<double, 3>, 3> ShieldAxesFromPose(const PoseSpec& pose) {
    const double norm = std::sqrt(
        pose.qw * pose.qw + pose.qx * pose.qx + pose.qy * pose.qy + pose.qz * pose.qz
    );
    if (norm <= 1.0e-12) {
        return {{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        }};
    }
    const double w = pose.qw / norm;
    const double x = pose.qx / norm;
    const double y = pose.qy / norm;
    const double z = pose.qz / norm;
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - z * w);
    const double r02 = 2.0 * (x * z + y * w);
    const double r10 = 2.0 * (x * y + z * w);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - x * w);
    const double r20 = 2.0 * (x * z - y * w);
    const double r21 = 2.0 * (y * z + x * w);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);
    return {{
        {r00, r10, r20},
        {r01, r11, r21},
        {r02, r12, r22}
    }};
}

void AddShieldAxisMetadata(
    SimulationResult& result,
    const std::string& prefix,
    const PoseSpec& pose
) {
    const auto axes = ShieldAxesFromPose(pose);
    const std::array<std::string, 3> names = {"x", "y", "z"};
    const std::array<std::string, 3> components = {"x", "y", "z"};
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        for (std::size_t component = 0; component < axes[axis].size(); ++component) {
            result.metadata[
                prefix + "_shield_axis_" + names[axis] + "_" + components[component]
            ] = std::to_string(axes[axis][component]);
        }
    }
}

int SignWithTolerance(const double value, const double tolerance = 1.0e-6) {
    if (std::abs(value) < tolerance) {
        return 0;
    }
    return value > 0.0 ? 1 : -1;
}

bool ShieldBlocksSourceDirection(
    const SourceSpec& source,
    const PoseSpec& detector_pose,
    const PoseSpec& shield_pose
) {
    const std::array<double, 3> direction = {
        source.x - detector_pose.x,
        source.y - detector_pose.y,
        source.z - detector_pose.z,
    };
    const auto normal = ShieldNormalFromPose(shield_pose);
    for (std::size_t idx = 0; idx < 3; ++idx) {
        if (SignWithTolerance(direction[idx]) != SignWithTolerance(normal[idx])) {
            return false;
        }
    }
    return true;
}

double TheoryTvlTransmission(
    const SourceSpec& source,
    const SceneSpec& scene,
    const RequestSpec& request
) {
    double exponent = 0.0;
    if (ShieldBlocksSourceDirection(source, request.detector_pose, request.fe_pose)) {
        const double thickness_cm = std::max(0.0, scene.fe_shield.thickness_m * 100.0);
        exponent += MuFromTvlMm(TvlMmForShield(source.isotope, "fe")) * thickness_cm;
    }
    if (ShieldBlocksSourceDirection(source, request.detector_pose, request.pb_pose)) {
        const double thickness_cm = std::max(0.0, scene.pb_shield.thickness_m * 100.0);
        exponent += MuFromTvlMm(TvlMmForShield(source.isotope, "pb")) * thickness_cm;
    }
    return std::exp(-exponent);
}

std::map<std::string, std::map<std::string, double>> PresetCompositionByMass() {
    return {
        {"air", {{"N", 0.755}, {"O", 0.232}, {"Ar", 0.013}}},
        {"water", {{"H", 0.1119}, {"O", 0.8881}}},
        {"concrete", {{"O", 0.525}, {"Si", 0.325}, {"Ca", 0.090}, {"Al", 0.060}}},
        {"aluminum", {{"Al", 1.0}}},
        {"iron", {{"Fe", 1.0}}},
        {"lead", {{"Pb", 1.0}}},
        {"steel", {{"Fe", 0.98}, {"C", 0.02}}},
        {"stainless_steel", {{"Fe", 0.70}, {"Cr", 0.19}, {"Ni", 0.10}, {"C", 0.01}}},
        {"cebr3", {{"Ce", 0.455}, {"Br", 0.545}}},
    };
}

double PresetDensity(const std::string& name) {
    if (name == "air") {
        return 0.001225;
    }
    if (name == "water") {
        return 1.0;
    }
    if (name == "concrete") {
        return 2.3;
    }
    if (name == "aluminum") {
        return 2.7;
    }
    if (name == "iron" || name == "fe") {
        return 7.87;
    }
    if (name == "lead" || name == "pb") {
        return 11.34;
    }
    if (name == "steel") {
        return 7.85;
    }
    if (name == "stainless_steel") {
        return 8.0;
    }
    if (name == "cebr3") {
        return 5.1;
    }
    return 1.0;
}

std::string NormalizeMaterialName(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "fe") {
        return "iron";
    }
    if (value == "pb") {
        return "lead";
    }
    if (value == "alu" || value == "aluminium") {
        return "aluminum";
    }
    return value;
}

std::string EnergyMetadataToken(const double energy_keV) {
    std::ostringstream stream;
    stream << "e" << std::fixed << std::setprecision(1) << energy_keV;
    std::string token = stream.str();
    for (char& ch : token) {
        if (ch == '.') {
            ch = 'p';
        }
    }
    return token;
}

std::string MaterialSpecName(const MaterialSpec& material) {
    const std::string raw = material.name.empty() ? material.preset_name : material.name;
    return NormalizeMaterialName(raw.empty() ? "air" : raw);
}

std::string CompositionCacheKey(const std::map<std::string, double>& composition_by_mass) {
    std::ostringstream stream;
    bool first = true;
    for (const auto& item : composition_by_mass) {
        if (!first) {
            stream << ";";
        }
        first = false;
        stream << item.first << ":" << std::setprecision(12) << item.second;
    }
    return stream.str();
}

G4Material* ResolveAttenuationMaterial(
    const MaterialSpec& material,
    const std::string& fallback_name
) {
    const std::string normalized_name = NormalizeMaterialName(
        MaterialSpecName(material).empty() ? fallback_name : MaterialSpecName(material)
    );
    auto* nist = G4NistManager::Instance();
    if (normalized_name == "air") {
        return nist->FindOrBuildMaterial("G4_AIR");
    }
    if (normalized_name == "concrete") {
        return nist->FindOrBuildMaterial("G4_CONCRETE");
    }
    if (normalized_name == "aluminum") {
        return nist->FindOrBuildMaterial("G4_Al");
    }
    if (normalized_name == "iron") {
        return nist->FindOrBuildMaterial("G4_Fe");
    }
    if (normalized_name == "lead") {
        return nist->FindOrBuildMaterial("G4_Pb");
    }
    if (normalized_name == "water") {
        return nist->FindOrBuildMaterial("G4_WATER");
    }

    std::map<std::string, double> composition = material.composition_by_mass;
    if (composition.empty()) {
        const auto presets = PresetCompositionByMass();
        const auto preset_it = presets.find(normalized_name);
        if (preset_it != presets.end()) {
            composition = preset_it->second;
        }
    }
    if (composition.empty()) {
        return nist->FindOrBuildMaterial("G4_AIR");
    }

    const double density_g_cm3 = material.density_g_cm3 > 0.0
        ? material.density_g_cm3
        : PresetDensity(normalized_name);
    const std::string key = normalized_name
        + "|rho=" + std::to_string(density_g_cm3)
        + "|comp=" + CompositionCacheKey(composition);
    static std::mutex cache_mutex;
    static std::map<std::string, G4Material*> cache;
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto cached = cache.find(key);
    if (cached != cache.end()) {
        return cached->second;
    }
    auto* resolved = new G4Material(
        "Attenuation_" + SanitizeMetadataToken(normalized_name)
            + "_" + std::to_string(cache.size()),
        density_g_cm3 * g / cm3,
        static_cast<G4int>(composition.size())
    );
    for (const auto& item : composition) {
        resolved->AddElement(nist->FindOrBuildElement(item.first), item.second);
    }
    cache[key] = resolved;
    return resolved;
}

double Geant4GammaMuCmInv(G4Material* material, const double energy_keV) {
    if (material == nullptr || energy_keV <= 0.0) {
        return 0.0;
    }
    G4EmCalculator calculator;
    const double length = calculator.ComputeGammaAttenuationLength(
        energy_keV * keV,
        material
    );
    if (!std::isfinite(length) || length <= 0.0) {
        return 0.0;
    }
    const double length_cm = length / cm;
    if (!std::isfinite(length_cm) || length_cm <= 0.0) {
        return 0.0;
    }
    return 1.0 / length_cm;
}

void AddMaterialMuMetadata(
    SimulationResult& result,
    const std::string& material_token,
    G4Material* material,
    const std::set<std::string>& isotopes
) {
    const std::string clean_material = SanitizeMetadataToken(material_token);
    for (const auto& isotope : isotopes) {
        for (const auto& line : GammaLinesForIsotope(isotope)) {
            const std::string energy_token = EnergyMetadataToken(line.energy_keV);
            const double mu_cm_inv = Geant4GammaMuCmInv(material, line.energy_keV);
            const std::string base = "geant4_mu_cm_inv_"
                + clean_material + "_" + SanitizeMetadataToken(isotope)
                + "_" + energy_token;
            result.metadata[base] = std::to_string(mu_cm_inv);
        }
    }
}

void AddGeant4MaterialMuMetadata(SimulationResult& result, const SceneSpec& scene) {
    std::set<std::string> isotopes;
    for (const auto& source : scene.sources) {
        if (!source.isotope.empty()) {
            isotopes.insert(source.isotope);
        }
    }
    if (isotopes.empty()) {
        return;
    }
    AddMaterialMuMetadata(
        result,
        "shield_fe",
        ResolveAttenuationMaterial(scene.fe_shield.material, "iron"),
        isotopes
    );
    AddMaterialMuMetadata(
        result,
        "shield_pb",
        ResolveAttenuationMaterial(scene.pb_shield.material, "lead"),
        isotopes
    );
    std::set<std::string> emitted_materials;
    for (const auto& volume : scene.volumes) {
        if (volume.transport_mode != "geant4") {
            continue;
        }
        const std::string material_name = MaterialSpecName(volume.material);
        if (material_name.empty() || emitted_materials.count(material_name) > 0) {
            continue;
        }
        emitted_materials.insert(material_name);
        AddMaterialMuMetadata(
            result,
            "obstacle_" + material_name,
            ResolveAttenuationMaterial(volume.material, material_name),
            isotopes
        );
    }
}

class TransportDiagnostics {
public:
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        total_track_steps_ = 0;
        detector_hit_steps_ = 0;
        secondary_count_ = 0;
        killed_non_gamma_secondary_count_ = 0;
        process_counts_.clear();
        volume_step_counts_.clear();
        interacted_gamma_track_ids_by_thread_.clear();
    }

    void BeginEvent() {
        std::lock_guard<std::mutex> lock(mutex_);
        interacted_gamma_track_ids_by_thread_[std::this_thread::get_id()].clear();
    }

    void AddStep(const G4Step* step) {
        if (step == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        total_track_steps_ += 1;
        const auto* pre_point = step->GetPreStepPoint();
        const auto* post_point = step->GetPostStepPoint();
        const auto* volume = pre_point == nullptr ? nullptr : pre_point->GetPhysicalVolume();
        const std::string volume_name = volume == nullptr ? "WorldBoundary" : volume->GetName();
        volume_step_counts_[volume_name] += 1;
        const auto* process = post_point == nullptr ? nullptr : post_point->GetProcessDefinedStep();
        const std::string process_name = process == nullptr ? "unknown" : process->GetProcessName();
        process_counts_[process_name] += 1;
        const auto* track = step->GetTrack();
        const std::string process_key = ToLower(process_name);
        if (
            track != nullptr
            && track->GetDefinition() == G4Gamma::Definition()
            && process_key != "transportation"
            && process_key != "unknown"
        ) {
            interacted_gamma_track_ids_by_thread_[std::this_thread::get_id()].insert(track->GetTrackID());
        }
        const auto* secondaries = step->GetSecondaryInCurrentStep();
        if (secondaries != nullptr) {
            secondary_count_ += static_cast<long long>(secondaries->size());
        }
    }

    void AddDetectorHitStep() {
        std::lock_guard<std::mutex> lock(mutex_);
        detector_hit_steps_ += 1;
    }

    void AddKilledNonGammaSecondary() {
        std::lock_guard<std::mutex> lock(mutex_);
        killed_non_gamma_secondary_count_ += 1;
    }

    long long TotalTrackSteps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_track_steps_;
    }

    long long DetectorHitSteps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return detector_hit_steps_;
    }

    long long SecondaryCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return secondary_count_;
    }

    long long KilledNonGammaSecondaryCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return killed_non_gamma_secondary_count_;
    }

    std::map<std::string, long long> ProcessCounts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return process_counts_;
    }

    std::map<std::string, long long> VolumeStepCounts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return volume_step_counts_;
    }

    long long ProcessCountForAliases(const std::set<std::string>& aliases) const {
        std::lock_guard<std::mutex> lock(mutex_);
        long long total = 0;
        for (const auto& item : process_counts_) {
            if (aliases.count(ToLower(item.first)) > 0) {
                total += item.second;
            }
        }
        return total;
    }

    bool TrackHadGammaInteraction(const G4Track* track) const {
        if (track == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto thread_it = interacted_gamma_track_ids_by_thread_.find(std::this_thread::get_id());
        if (thread_it == interacted_gamma_track_ids_by_thread_.end()) {
            return false;
        }
        return thread_it->second.count(track->GetTrackID()) > 0;
    }

private:
    mutable std::mutex mutex_;
    long long total_track_steps_ = 0;
    long long detector_hit_steps_ = 0;
    long long secondary_count_ = 0;
    long long killed_non_gamma_secondary_count_ = 0;
    std::map<std::string, long long> process_counts_;
    std::map<std::string, long long> volume_step_counts_;
    std::map<std::thread::id, std::set<int>> interacted_gamma_track_ids_by_thread_;
};

class EventStore {
public:
    void BeginEvent() {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto thread_id = std::this_thread::get_id();
        current_edep_mev_by_thread_[thread_id] = 0.0;
        current_weight_by_thread_[thread_id] = 1.0;
        current_entry_class_by_thread_[thread_id] = DetectorEntryClass::kUncollidedPrimary;
    }

    void AddEnergyDeposit(
        const double edep_mev,
        const double track_weight,
        const DetectorEntryClass entry_class
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto thread_id = std::this_thread::get_id();
        current_edep_mev_by_thread_[thread_id] += edep_mev;
        if (std::isfinite(track_weight) && track_weight > 0.0) {
            current_weight_by_thread_[thread_id] = track_weight;
        }
        current_entry_class_by_thread_[thread_id] = MergeDetectorEntryClass(
            current_entry_class_by_thread_[thread_id],
            entry_class
        );
    }

    void EndEvent() {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto thread_id = std::this_thread::get_id();
        const auto it = current_edep_mev_by_thread_.find(thread_id);
        if (it == current_edep_mev_by_thread_.end()) {
            return;
        }
        if (it->second > 0.0) {
            const auto weight_it = current_weight_by_thread_.find(thread_id);
            const double weight = weight_it == current_weight_by_thread_.end()
                ? 1.0
                : std::max(0.0, weight_it->second);
            const auto entry_it = current_entry_class_by_thread_.find(thread_id);
            const auto entry_class = entry_it == current_entry_class_by_thread_.end()
                ? DetectorEntryClass::kUncollidedPrimary
                : entry_it->second;
            event_edep_mev_.push_back({it->second, weight, entry_class});
        }
        current_edep_mev_by_thread_.erase(it);
        current_weight_by_thread_.erase(thread_id);
        current_entry_class_by_thread_.erase(thread_id);
    }

    const std::vector<WeightedEventDeposit>& EventDepositsMeV() const {
        return event_edep_mev_;
    }

    void ClearDeposits() {
        std::lock_guard<std::mutex> lock(mutex_);
        event_edep_mev_.clear();
        current_edep_mev_by_thread_.clear();
        current_weight_by_thread_.clear();
        current_entry_class_by_thread_.clear();
    }

private:
    mutable std::mutex mutex_;
    std::map<std::thread::id, double> current_edep_mev_by_thread_;
    std::map<std::thread::id, double> current_weight_by_thread_;
    std::map<std::thread::id, DetectorEntryClass> current_entry_class_by_thread_;
    std::vector<WeightedEventDeposit> event_edep_mev_;
};

DetectorEntryClass ClassifyDetectorEntryTrack(
    const G4Track* track,
    const TransportDiagnostics* diagnostics
) {
    if (track == nullptr || track->GetDefinition() != G4Gamma::Definition()) {
        return DetectorEntryClass::kInteractedPrimary;
    }
    if (track->GetParentID() > 0) {
        return DetectorEntryClass::kSecondary;
    }
    if (diagnostics != nullptr && diagnostics->TrackHadGammaInteraction(track)) {
        return DetectorEntryClass::kInteractedPrimary;
    }
    return DetectorEntryClass::kUncollidedPrimary;
}

class CrystalSensitiveDetector : public G4VSensitiveDetector {
public:
    CrystalSensitiveDetector(
        EventStore* store,
        TransportDiagnostics* diagnostics,
        const std::string& detector_scoring_mode
    ) : G4VSensitiveDetector("CrystalSD"),
        store_(store),
        diagnostics_(diagnostics),
        score_energy_deposits_(NormalizeDetectorScoringMode(detector_scoring_mode) == "full_transport") {}

    void Initialize(G4HCofThisEvent*) override {
        store_->BeginEvent();
        if (diagnostics_ != nullptr) {
            diagnostics_->BeginEvent();
        }
    }

    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        if (step == nullptr) {
            return false;
        }
        if (!score_energy_deposits_) {
            return false;
        }
        const auto* track = step->GetTrack();
        const double weight = track == nullptr ? 1.0 : track->GetWeight();
        const double edep_mev = step->GetTotalEnergyDeposit() / MeV;
        if (edep_mev > 0.0 && diagnostics_ != nullptr) {
            diagnostics_->AddDetectorHitStep();
        }
        store_->AddEnergyDeposit(edep_mev, weight, ClassifyDetectorEntryTrack(track, diagnostics_));
        return true;
    }

    void EndOfEvent(G4HCofThisEvent*) override {
        store_->EndEvent();
    }

private:
    EventStore* store_ = nullptr;
    TransportDiagnostics* diagnostics_ = nullptr;
    bool score_energy_deposits_ = true;
};

class Geant4SceneConstruction : public G4VUserDetectorConstruction {
public:
    Geant4SceneConstruction(
        const SceneSpec* scene,
        const RequestSpec* request,
        EventStore* event_store,
        TransportDiagnostics* diagnostics,
        std::string detector_scoring_mode,
        const bool place_shields
    ) : scene_(scene),
        request_(request),
        event_store_(event_store),
        diagnostics_(diagnostics),
        detector_scoring_mode_(std::move(detector_scoring_mode)),
        place_shields_(place_shields) {}

    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* world_material = ResolveMaterial("air", -1.0, {}, "air");
        const double world_x = 0.5 * std::max(40.0, scene_->room_x + 20.0) * m;
        const double world_y = 0.5 * std::max(40.0, scene_->room_y + 20.0) * m;
        const double world_z = 0.5 * std::max(20.0, scene_->room_z + 10.0) * m;
        auto* world_solid = new G4Box("World", world_x, world_y, world_z);
        auto* world_logic = new G4LogicalVolume(world_solid, world_material, "WorldLV");
        auto* world_physical = new G4PVPlacement(
            nullptr,
            G4ThreeVector(),
            world_logic,
            "WorldPV",
            nullptr,
            false,
            0,
            true
        );
        for (std::size_t index = 0; index < scene_->volumes.size(); ++index) {
            const auto& volume = scene_->volumes[index];
            auto* material = ResolveMaterial(
                volume.material.name,
                volume.material.density_g_cm3,
                volume.material.composition_by_mass,
                volume.material.preset_name
            );
            auto* solid = BuildSolid(volume, "StaticSolid_" + std::to_string(index));
            auto* logic = new G4LogicalVolume(solid, material, "StaticLV_" + std::to_string(index));
            logic->SetVisAttributes(G4VisAttributes::GetInvisible());
            auto rotation = QuaternionToPlacementRotation(volume.qw, volume.qx, volume.qy, volume.qz);
            G4ThreeVector placement(volume.tx * m, volume.ty * m, volume.tz * m);
            if (volume.shape == "mesh") {
                rotation = std::make_unique<G4RotationMatrix>();
                placement = G4ThreeVector();
            }
            new G4PVPlacement(
                rotation.release(),
                placement,
                logic,
                volume.path,
                world_logic,
                false,
                static_cast<int>(index),
                true
            );
        }
        if (place_shields_) {
            fe_shield_physical_ = BuildShield(scene_->fe_shield, request_->fe_pose, world_logic, 1001);
            pb_shield_physical_ = BuildShield(scene_->pb_shield, request_->pb_pose, world_logic, 1002);
        }
        BuildDetector(world_logic, nist);
        return world_physical;
    }

    void UpdateShieldPoses(const RequestSpec& request) {
        if (!place_shields_) {
            return;
        }
        UpdatePhysicalPose(fe_shield_physical_, request.fe_pose);
        UpdatePhysicalPose(pb_shield_physical_, request.pb_pose);
        if (auto* run_manager = G4RunManager::GetRunManager()) {
            run_manager->GeometryHasBeenModified();
        }
    }

    void ConstructSDandField() override {
        auto* sd_manager = G4SDManager::GetSDMpointer();
        auto* crystal_sd = new CrystalSensitiveDetector(
            event_store_,
            diagnostics_,
            detector_scoring_mode_
        );
        sd_manager->AddNewDetector(crystal_sd);
        SetSensitiveDetector("DetectorCrystalLV", crystal_sd);
    }

private:
    G4VSolid* BuildSolid(const VolumeSpec& volume, const std::string& name) const {
        if (volume.shape == "box") {
            return new G4Box(name, 0.5 * volume.sx * m, 0.5 * volume.sy * m, 0.5 * volume.sz * m);
        }
        if (volume.shape == "sphere") {
            return new G4Sphere(name, 0.0, volume.radius_m * m, 0.0, 360.0 * deg, 0.0, 180.0 * deg);
        }
        if (volume.shape == "mesh") {
            auto* solid = new G4TessellatedSolid(name);
            for (const auto& triangle : volume.triangles) {
                auto* facet = new G4TriangularFacet(
                    G4ThreeVector(triangle[0] * m, triangle[1] * m, triangle[2] * m),
                    G4ThreeVector(triangle[3] * m, triangle[4] * m, triangle[5] * m),
                    G4ThreeVector(triangle[6] * m, triangle[7] * m, triangle[8] * m),
                    ABSOLUTE
                );
                solid->AddFacet(facet);
            }
            solid->SetSolidClosed(true);
            return solid;
        }
        throw std::runtime_error("Unsupported volume shape: " + volume.shape);
    }

    G4VPhysicalVolume* BuildShield(
        const ShieldSpec& shield,
        const PoseSpec& pose,
        G4LogicalVolume* parent_logic,
        int copy_number
    ) {
        auto* material = ResolveMaterial(
            shield.material.name,
            shield.material.density_g_cm3,
            shield.material.composition_by_mass,
            shield.material.preset_name
        );
        G4VSolid* solid = nullptr;
        if (shield.shape == "spherical_octant_shell") {
            const double inner_radius_m = std::max(0.0, shield.inner_radius_m);
            const double outer_radius_m = std::max(inner_radius_m + 1.0e-6, shield.outer_radius_m);
            solid = new G4Sphere(
                shield.kind + "_ShieldSolid",
                inner_radius_m * m,
                outer_radius_m * m,
                0.0 * deg,
                90.0 * deg,
                0.0 * deg,
                90.0 * deg
            );
        } else {
            solid = new G4Box(
                shield.kind + "_ShieldSolid",
                0.5 * shield.sx * m,
                0.5 * shield.sy * m,
                0.5 * shield.sz * m
            );
        }
        auto* logic = new G4LogicalVolume(solid, material, shield.kind + "_ShieldLV");
        auto rotation = QuaternionToPlacementRotation(pose.qw, pose.qx, pose.qy, pose.qz);
        return new G4PVPlacement(
            rotation.release(),
            G4ThreeVector(pose.x * m, pose.y * m, pose.z * m),
            logic,
            shield.path,
            parent_logic,
            false,
            copy_number,
            true
        );
    }

    void UpdatePhysicalPose(G4VPhysicalVolume* physical, const PoseSpec& pose) const {
        if (physical == nullptr) {
            return;
        }
        auto rotation = QuaternionToPlacementRotation(pose.qw, pose.qx, pose.qy, pose.qz);
        physical->SetTranslation(G4ThreeVector(pose.x * m, pose.y * m, pose.z * m));
        physical->SetRotation(rotation.release());
    }

    void BuildDetector(G4LogicalVolume* parent_logic, G4NistManager*) {
        const auto& detector = scene_->detector;
        auto* housing_material = ResolveMaterial(detector.housing_material, -1.0, {}, detector.housing_material);
        auto* crystal_material = ResolveMaterial(detector.crystal_material, -1.0, {}, detector.crystal_material);
        const double outer_radius_m = detector.crystal_radius_m + detector.housing_thickness_m;
        auto* housing_solid = new G4Sphere(
            "DetectorHousingSolid",
            0.0,
            outer_radius_m * m,
            0.0,
            360.0 * deg,
            0.0,
            180.0 * deg
        );
        auto* housing_logic = new G4LogicalVolume(housing_solid, housing_material, "DetectorHousingLV");
        auto housing_rotation = QuaternionToPlacementRotation(
            request_->detector_pose.qw,
            request_->detector_pose.qx,
            request_->detector_pose.qy,
            request_->detector_pose.qz
        );
        new G4PVPlacement(
            housing_rotation.release(),
            G4ThreeVector(
                request_->detector_pose.x * m,
                request_->detector_pose.y * m,
                request_->detector_pose.z * m
            ),
            housing_logic,
            "DetectorHousingPV",
            parent_logic,
            false,
            2001,
            true
        );
        auto* crystal_solid = new G4Sphere(
            "DetectorCrystalSolid",
            0.0,
            detector.crystal_radius_m * m,
            0.0,
            360.0 * deg,
            0.0,
            180.0 * deg
        );
        auto* crystal_logic = new G4LogicalVolume(crystal_solid, crystal_material, "DetectorCrystalLV");
        new G4PVPlacement(
            nullptr,
            G4ThreeVector(0.0, 0.0, 0.0),
            crystal_logic,
            "DetectorCrystalPV",
            housing_logic,
            false,
            2002,
            true
        );
    }

    G4Material* ResolveMaterial(
        const std::string& material_name,
        const double density_g_cm3,
        std::map<std::string, double> composition_by_mass,
        const std::string& preset_name
    ) const {
        const auto normalized_name = NormalizeMaterialName(material_name.empty() ? preset_name : material_name);
        auto* nist = G4NistManager::Instance();
        if (normalized_name == "air") {
            return nist->FindOrBuildMaterial("G4_AIR");
        }
        if (normalized_name == "concrete") {
            return nist->FindOrBuildMaterial("G4_CONCRETE");
        }
        if (normalized_name == "aluminum") {
            return nist->FindOrBuildMaterial("G4_Al");
        }
        if (normalized_name == "iron") {
            return nist->FindOrBuildMaterial("G4_Fe");
        }
        if (normalized_name == "lead") {
            return nist->FindOrBuildMaterial("G4_Pb");
        }
        if (normalized_name == "water") {
            return nist->FindOrBuildMaterial("G4_WATER");
        }
        if (composition_by_mass.empty()) {
            const auto presets = PresetCompositionByMass();
            const auto preset_it = presets.find(normalized_name);
            if (preset_it != presets.end()) {
                composition_by_mass = preset_it->second;
            }
        }
        if (composition_by_mass.empty()) {
            return nist->FindOrBuildMaterial("G4_AIR");
        }
        const double density = (density_g_cm3 > 0.0 ? density_g_cm3 : PresetDensity(normalized_name)) * g / cm3;
        auto* material = new G4Material(
            "Custom_" + normalized_name + "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + "_" + std::to_string(material_counter_++),
            density,
            static_cast<G4int>(composition_by_mass.size())
        );
        for (const auto& item : composition_by_mass) {
            material->AddElement(nist->FindOrBuildElement(item.first), item.second);
        }
        return material;
    }

    std::unique_ptr<G4RotationMatrix> QuaternionToRotation(
        const double qw,
        const double qx,
        const double qy,
        const double qz
    ) const {
        auto rotation = std::make_unique<G4RotationMatrix>();
        const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (norm <= 1.0e-12) {
            return rotation;
        }
        const double w = qw / norm;
        const double x = qx / norm;
        const double y = qy / norm;
        const double z = qz / norm;
        const double r00 = 1.0 - 2.0 * (y * y + z * z);
        const double r01 = 2.0 * (x * y - z * w);
        const double r02 = 2.0 * (x * z + y * w);
        const double r10 = 2.0 * (x * y + z * w);
        const double r11 = 1.0 - 2.0 * (x * x + z * z);
        const double r12 = 2.0 * (y * z - x * w);
        const double r20 = 2.0 * (x * z - y * w);
        const double r21 = 2.0 * (y * z + x * w);
        const double r22 = 1.0 - 2.0 * (x * x + y * y);
        rotation->setRows(
            G4ThreeVector(r00, r01, r02),
            G4ThreeVector(r10, r11, r12),
            G4ThreeVector(r20, r21, r22)
        );
        return rotation;
    }

    std::unique_ptr<G4RotationMatrix> QuaternionToPlacementRotation(
        const double qw,
        const double qx,
        const double qy,
        const double qz
    ) const {
        auto rotation = QuaternionToRotation(qw, qx, qy, qz);
        // G4PVPlacement expects the daughter-to-mother transform inverse of the
        // active quaternion exported from Isaac/PF geometry.
        rotation->invert();
        return rotation;
    }

    const SceneSpec* scene_ = nullptr;
    const RequestSpec* request_ = nullptr;
    EventStore* event_store_ = nullptr;
    TransportDiagnostics* diagnostics_ = nullptr;
    std::string detector_scoring_mode_ = "full_transport";
    bool place_shields_ = true;
    G4VPhysicalVolume* fe_shield_physical_ = nullptr;
    G4VPhysicalVolume* pb_shield_physical_ = nullptr;
    mutable int material_counter_ = 0;
};

struct PrimarySourceSnapshot {
    G4ThreeVector position;
    double energy_keV = 662.0;
    std::string source_bias_mode = "analog";
    std::string source_rate_model = "detector_cps_1m";
    G4ThreeVector detector_center;
    double cone_half_angle_rad = CLHEP::pi;
    double isotropic_fraction = 1.0;
    double history_weight = 1.0;
};

struct PrimaryDirectionSample {
    G4ThreeVector direction;
    double weight = 1.0;
};

class PrimarySourceState {
public:
    void Configure(
        const G4ThreeVector& position,
        const double energy_keV,
        const std::string& source_bias_mode,
        const std::string& source_rate_model,
        const G4ThreeVector& detector_center,
        const double cone_half_angle_rad,
        const double isotropic_fraction,
        const double history_weight
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        position_ = position;
        energy_keV_ = energy_keV;
        source_bias_mode_ = NormalizeSourceBiasMode(source_bias_mode);
        source_rate_model_ = NormalizeSourceRateModel(source_rate_model);
        detector_center_ = detector_center;
        cone_half_angle_rad_ = std::clamp(cone_half_angle_rad, 1.0e-9, CLHEP::pi);
        isotropic_fraction_ = std::clamp(isotropic_fraction, 1.0e-6, 1.0);
        history_weight_ = std::isfinite(history_weight) && history_weight > 0.0 ? history_weight : 1.0;
    }

    PrimarySourceSnapshot Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            position_,
            energy_keV_,
            source_bias_mode_,
            source_rate_model_,
            detector_center_,
            cone_half_angle_rad_,
            isotropic_fraction_,
            history_weight_
        };
    }

private:
    mutable std::mutex mutex_;
    G4ThreeVector position_ = G4ThreeVector();
    double energy_keV_ = 662.0;
    std::string source_bias_mode_ = "analog";
    std::string source_rate_model_ = "detector_cps_1m";
    G4ThreeVector detector_center_ = G4ThreeVector();
    double cone_half_angle_rad_ = CLHEP::pi;
    double isotropic_fraction_ = 1.0;
    double history_weight_ = 1.0;
};

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    explicit PrimaryGeneratorAction(const PrimarySourceState* state)
        : state_(state) {
        particle_gun_ = std::make_unique<G4ParticleGun>(1);
        particle_gun_->SetParticleDefinition(G4Gamma::Definition());
    }

    void GeneratePrimaries(G4Event* event) override {
        const auto source = state_->Snapshot();
        const auto sample = SampleDirection(source);
        particle_gun_->SetParticlePosition(source.position);
        particle_gun_->SetParticleEnergy(source.energy_keV * keV);
        particle_gun_->SetParticleMomentumDirection(sample.direction);
        particle_gun_->GeneratePrimaryVertex(event);
        ApplyPrimaryWeight(event, sample.weight * source.history_weight);
    }

private:
    void ApplyPrimaryWeight(G4Event* event, const double weight) const {
        if (event == nullptr) {
            return;
        }
        const int vertex_count = event->GetNumberOfPrimaryVertex();
        if (vertex_count <= 0) {
            return;
        }
        auto* vertex = event->GetPrimaryVertex(vertex_count - 1);
        if (vertex == nullptr) {
            return;
        }
        auto* particle = vertex->GetPrimary();
        while (particle != nullptr) {
            particle->SetWeight(weight);
            particle = particle->GetNext();
        }
    }

    PrimaryDirectionSample SampleDirection(const PrimarySourceSnapshot& source) const {
        if (NormalizeSourceBiasMode(source.source_bias_mode) == "detector_cone") {
            const G4ThreeVector axis_vector = source.detector_center - source.position;
            if (axis_vector.mag2() <= 1.0e-18) {
                return {RandomIsotropicDirection(), 1.0};
            }
            const G4ThreeVector axis = axis_vector.unit();
            const double theta = std::clamp(source.cone_half_angle_rad, 1.0e-9, CLHEP::pi);
            return {RandomConeDirection(axis, std::cos(theta)), 1.0};
        }
        if (NormalizeSourceBiasMode(source.source_bias_mode) != "mixture_cone_isotropic") {
            return {RandomIsotropicDirection(), 1.0};
        }
        const double f_iso = std::clamp(source.isotropic_fraction, 1.0e-6, 1.0);
        if (f_iso >= 1.0 - 1.0e-12) {
            return {RandomIsotropicDirection(), 1.0};
        }
        const G4ThreeVector axis_vector = source.detector_center - source.position;
        if (axis_vector.mag2() <= 1.0e-18) {
            return {RandomIsotropicDirection(), 1.0};
        }
        const G4ThreeVector axis = axis_vector.unit();
        const double theta = std::clamp(source.cone_half_angle_rad, 1.0e-9, CLHEP::pi);
        const double cos_theta = std::cos(theta);
        const bool sample_isotropic = G4UniformRand() < f_iso;
        const G4ThreeVector direction = sample_isotropic
            ? RandomIsotropicDirection()
            : RandomConeDirection(axis, cos_theta);
        const bool in_cone = direction.dot(axis) >= cos_theta - 1.0e-12;
        const double iso_pdf = 1.0 / (4.0 * CLHEP::pi);
        const double cone_pdf = 1.0 / ConeSolidAngleSr(theta);
        const double sample_pdf = f_iso * iso_pdf + (in_cone ? (1.0 - f_iso) * cone_pdf : 0.0);
        const double weight = sample_pdf > 0.0 ? iso_pdf / sample_pdf : 1.0;
        return {direction, std::max(0.0, weight)};
    }

    G4ThreeVector RandomIsotropicDirection() const {
        const double u = 2.0 * G4UniformRand() - 1.0;
        const double phi = 2.0 * CLHEP::pi * G4UniformRand();
        const double scale = std::sqrt(std::max(0.0, 1.0 - u * u));
        return G4ThreeVector(scale * std::cos(phi), scale * std::sin(phi), u);
    }

    G4ThreeVector RandomConeDirection(const G4ThreeVector& axis, const double cos_theta) const {
        const double u = cos_theta + (1.0 - cos_theta) * G4UniformRand();
        const double phi = 2.0 * CLHEP::pi * G4UniformRand();
        const double transverse = std::sqrt(std::max(0.0, 1.0 - u * u));
        const G4ThreeVector basis_x = axis.orthogonal().unit();
        const G4ThreeVector basis_y = axis.cross(basis_x).unit();
        return (u * axis + transverse * std::cos(phi) * basis_x + transverse * std::sin(phi) * basis_y).unit();
    }

    std::unique_ptr<G4ParticleGun> particle_gun_;
    const PrimarySourceState* state_ = nullptr;
};

class SecondaryTransportStackingAction : public G4UserStackingAction {
public:
    SecondaryTransportStackingAction(
        std::string secondary_transport_mode,
        std::string detector_scoring_mode,
        TransportDiagnostics* diagnostics,
        const G4ThreeVector& detector_center,
        const double detector_radius
    ) : secondary_transport_mode_(NormalizeSecondaryTransportMode(secondary_transport_mode)),
        detector_scoring_mode_(NormalizeDetectorScoringMode(detector_scoring_mode)),
        diagnostics_(diagnostics),
        detector_center_(detector_center),
        detector_radius_(std::max(0.0, detector_radius)) {}

    G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override {
        if (secondary_transport_mode_ != "gamma_only" || track == nullptr) {
            return fUrgent;
        }
        if (track->GetParentID() <= 0) {
            return fUrgent;
        }
        if (track->GetDefinition() == G4Gamma::Definition()) {
            return fUrgent;
        }
        if (
            detector_scoring_mode_ != "incident_gamma_energy"
            && IsInsideDetectorAssembly(track->GetPosition())
        ) {
            return fUrgent;
        }
        if (diagnostics_ != nullptr) {
            diagnostics_->AddKilledNonGammaSecondary();
        }
        return fKill;
    }

private:
    bool IsInsideDetectorAssembly(const G4ThreeVector& position) const {
        constexpr double kTolerance = 1.0 * mm;
        return (position - detector_center_).mag() <= detector_radius_ + kTolerance;
    }

    std::string secondary_transport_mode_ = "full_transport";
    std::string detector_scoring_mode_ = "full_transport";
    TransportDiagnostics* diagnostics_ = nullptr;
    G4ThreeVector detector_center_;
    double detector_radius_ = 0.0;
};

class TransportSteppingAction : public G4UserSteppingAction {
public:
    TransportSteppingAction(
        std::set<std::string> absorbing_volume_names,
        TransportDiagnostics* diagnostics,
        EventStore* event_store,
        const std::string& detector_scoring_mode,
        const std::string& secondary_transport_mode
    ) : absorbing_volume_names_(std::move(absorbing_volume_names)),
        diagnostics_(diagnostics),
        event_store_(event_store),
        detector_scoring_mode_(NormalizeDetectorScoringMode(detector_scoring_mode)),
        secondary_transport_mode_(NormalizeSecondaryTransportMode(secondary_transport_mode)) {}

    void UserSteppingAction(const G4Step* step) override {
        if (step == nullptr) {
            return;
        }
        if (diagnostics_ != nullptr) {
            diagnostics_->AddStep(step);
        }
        if (ScoreFastDetectorEntry(step)) {
            return;
        }
        if (KillNonGammaOutsideDetector(step)) {
            return;
        }
        if (absorbing_volume_names_.empty()) {
            return;
        }
        auto* track = step->GetTrack();
        if (track == nullptr) {
            return;
        }
        if (IsAbsorbingVolume(step->GetPreStepPoint()->GetPhysicalVolume())) {
            track->SetTrackStatus(fStopAndKill);
            return;
        }
        if (IsAbsorbingVolume(step->GetPostStepPoint()->GetPhysicalVolume())) {
            track->SetTrackStatus(fStopAndKill);
        }
    }

private:
    bool IsDetectorCrystalVolume(const G4VPhysicalVolume* volume) const {
        if (volume == nullptr) {
            return false;
        }
        return volume->GetName() == "DetectorCrystalPV";
    }

    bool IsDetectorVolume(const G4VPhysicalVolume* volume) const {
        if (volume == nullptr) {
            return false;
        }
        const auto name = volume->GetName();
        return name == "DetectorCrystalPV" || name == "DetectorHousingPV";
    }

    bool ScoreFastDetectorEntry(const G4Step* step) const {
        if (detector_scoring_mode_ != "incident_gamma_energy") {
            return false;
        }
        auto* track = step->GetTrack();
        if (track == nullptr) {
            return false;
        }
        const auto* pre_point = step->GetPreStepPoint();
        const auto* post_point = step->GetPostStepPoint();
        if (pre_point == nullptr || post_point == nullptr) {
            return false;
        }
        if (IsDetectorVolume(pre_point->GetPhysicalVolume())) {
            track->SetTrackStatus(fStopAndKill);
            return true;
        }
        if (!IsDetectorVolume(post_point->GetPhysicalVolume())) {
            return false;
        }
        if (track->GetDefinition() == G4Gamma::Definition()) {
            double energy_mev = post_point->GetKineticEnergy() / MeV;
            if (!(std::isfinite(energy_mev) && energy_mev > 0.0)) {
                energy_mev = track->GetKineticEnergy() / MeV;
            }
            if (std::isfinite(energy_mev) && energy_mev > 0.0 && event_store_ != nullptr) {
                event_store_->AddEnergyDeposit(
                    energy_mev,
                    std::max(0.0, track->GetWeight()),
                    ClassifyDetectorEntryTrack(track, diagnostics_)
                );
                if (diagnostics_ != nullptr) {
                    diagnostics_->AddDetectorHitStep();
                }
            }
        }
        track->SetTrackStatus(fStopAndKill);
        return true;
    }

    bool KillNonGammaOutsideDetector(const G4Step* step) const {
        if (secondary_transport_mode_ != "gamma_only") {
            return false;
        }
        auto* track = step->GetTrack();
        if (track == nullptr || track->GetParentID() <= 0) {
            return false;
        }
        if (track->GetDefinition() == G4Gamma::Definition()) {
            return false;
        }
        const auto* pre_point = step->GetPreStepPoint();
        if (
            detector_scoring_mode_ != "incident_gamma_energy"
            && pre_point != nullptr
            && IsDetectorVolume(pre_point->GetPhysicalVolume())
        ) {
            return false;
        }
        if (diagnostics_ != nullptr) {
            diagnostics_->AddKilledNonGammaSecondary();
        }
        track->SetTrackStatus(fStopAndKill);
        return true;
    }

    bool IsAbsorbingVolume(const G4VPhysicalVolume* volume) const {
        if (volume == nullptr) {
            return false;
        }
        return absorbing_volume_names_.count(volume->GetName()) > 0;
    }

    std::set<std::string> absorbing_volume_names_;
    TransportDiagnostics* diagnostics_ = nullptr;
    EventStore* event_store_ = nullptr;
    std::string detector_scoring_mode_ = "full_transport";
    std::string secondary_transport_mode_ = "full_transport";
};

class EventAction : public G4UserEventAction {
public:
    EventAction(EventStore* store, TransportDiagnostics* diagnostics)
        : store_(store), diagnostics_(diagnostics) {}

    void BeginOfEventAction(const G4Event*) override {
        if (store_ != nullptr) {
            store_->BeginEvent();
        }
        if (diagnostics_ != nullptr) {
            diagnostics_->BeginEvent();
        }
    }

    void EndOfEventAction(const G4Event*) override {
        if (store_ != nullptr) {
            store_->EndEvent();
        }
    }

private:
    EventStore* store_ = nullptr;
    TransportDiagnostics* diagnostics_ = nullptr;
};

class SidecarActionInitialization : public G4VUserActionInitialization {
public:
    SidecarActionInitialization(
        const PrimarySourceState* source_state,
        std::set<std::string> absorbing_volume_names,
        TransportDiagnostics* diagnostics,
        EventStore* event_store,
        std::string detector_scoring_mode,
        std::string secondary_transport_mode,
        G4ThreeVector detector_center,
        double detector_radius
    ) : source_state_(source_state),
        absorbing_volume_names_(std::move(absorbing_volume_names)),
        diagnostics_(diagnostics),
        event_store_(event_store),
        detector_scoring_mode_(std::move(detector_scoring_mode)),
        secondary_transport_mode_(std::move(secondary_transport_mode)),
        detector_center_(detector_center),
        detector_radius_(std::max(0.0, detector_radius)) {}

    void Build() const override {
        SetUserAction(new PrimaryGeneratorAction(source_state_));
        SetUserAction(new EventAction(event_store_, diagnostics_));
        SetUserAction(new SecondaryTransportStackingAction(
            secondary_transport_mode_,
            detector_scoring_mode_,
            diagnostics_,
            detector_center_,
            detector_radius_
        ));
        SetUserAction(new TransportSteppingAction(
            absorbing_volume_names_,
            diagnostics_,
            event_store_,
            detector_scoring_mode_,
            secondary_transport_mode_
        ));
    }

private:
    const PrimarySourceState* source_state_ = nullptr;
    std::set<std::string> absorbing_volume_names_;
    TransportDiagnostics* diagnostics_ = nullptr;
    EventStore* event_store_ = nullptr;
    std::string detector_scoring_mode_ = "full_transport";
    std::string secondary_transport_mode_ = "full_transport";
    G4ThreeVector detector_center_;
    double detector_radius_ = 0.0;
};

G4RunManager* CreateConfiguredRunManager(const int thread_count, bool* use_multithreaded) {
    const int requested_threads = std::max(1, thread_count);
    if (use_multithreaded != nullptr) {
        *use_multithreaded = false;
    }
#ifdef G4MULTITHREADED
    if (requested_threads > 1) {
        auto* run_manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::MTOnly);
        if (auto* mt_manager = dynamic_cast<G4MTRunManager*>(run_manager)) {
            mt_manager->SetNumberOfThreads(requested_threads);
            if (use_multithreaded != nullptr) {
                *use_multithreaded = true;
            }
        }
        return run_manager;
    }
#endif
    return G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);
}

void BeamOnHistories(G4RunManager* run_manager, const long histories) {
    long remaining = std::max(0L, histories);
    constexpr int kMaxBeamOnEvents = 1000000000;
    while (remaining > 0) {
        const int chunk = static_cast<int>(std::min<long>(remaining, kMaxBeamOnEvents));
        run_manager->BeamOn(chunk);
        remaining -= chunk;
    }
}

SceneSpec ReadSceneFile(const std::string& scene_path) {
    std::ifstream input(scene_path);
    if (!input) {
        throw std::runtime_error("Failed to open scene file: " + scene_path);
    }
    SceneSpec scene;
    std::unordered_map<std::string, std::size_t> volume_index_by_path;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto tokens = Split(line);
        if (tokens.empty()) {
            continue;
        }
        const auto fields = ParseFields(tokens);
        if (tokens[0] == "SCENE") {
            scene.scene_hash = ParseString(fields, "scene_hash");
            scene.usd_path = ParseString(fields, "usd_path");
            scene.room_x = ParseDouble(fields, "room_x", 10.0);
            scene.room_y = ParseDouble(fields, "room_y", 20.0);
            scene.room_z = ParseDouble(fields, "room_z", 10.0);
        } else if (tokens[0] == "DETECTOR") {
            scene.detector.crystal_radius_m = ParseDouble(
                fields,
                "crystal_radius_m",
                kDefaultCrystalRadiusM
            );
            scene.detector.crystal_length_m = ParseDouble(
                fields,
                "crystal_length_m",
                kDefaultCrystalLengthM
            );
            scene.detector.housing_thickness_m = ParseDouble(
                fields,
                "housing_thickness_m",
                kDefaultHousingThicknessM
            );
            scene.detector.crystal_shape = ParseString(fields, "crystal_shape", "sphere");
            scene.detector.crystal_material = ParseString(fields, "crystal_material", "cebr3");
            scene.detector.housing_material = ParseString(fields, "housing_material", "aluminum");
        } else if (tokens[0] == "SHIELD") {
            const auto shield_kind = ParseString(fields, "kind");
            ShieldSpec shield = shield_kind == "pb" ? DefaultPbShieldSpec() : DefaultFeShieldSpec();
            shield.kind = shield_kind;
            shield.path = ParseString(fields, "path");
            shield.shape = ParseString(fields, "shape", "spherical_octant_shell");
            shield.inner_radius_m = ParseDouble(fields, "inner_radius_m", shield.inner_radius_m);
            shield.outer_radius_m = ParseDouble(fields, "outer_radius_m", shield.outer_radius_m);
            shield.thickness_m = ParseDouble(fields, "thickness_m", shield.thickness_m);
            if (shield.outer_radius_m <= shield.inner_radius_m && shield.thickness_m > 0.0) {
                shield.outer_radius_m = shield.inner_radius_m + shield.thickness_m;
            }
            shield.sx = ParseDouble(fields, "sx", 0.25);
            shield.sy = ParseDouble(fields, "sy", 0.08);
            shield.sz = ParseDouble(fields, "sz", 0.25);
            shield.material.name = ParseString(fields, "material_name");
            shield.material.density_g_cm3 = ParseDouble(fields, "density_g_cm3", -1.0);
            shield.material.preset_name = ParseString(fields, "preset_name");
            if (shield.kind == "fe") {
                scene.fe_shield = shield;
            } else if (shield.kind == "pb") {
                scene.pb_shield = shield;
            }
        } else if (tokens[0] == "SOURCE") {
            SourceSpec source;
            source.isotope = ParseString(fields, "isotope");
            source.x = ParseDouble(fields, "x");
            source.y = ParseDouble(fields, "y");
            source.z = ParseDouble(fields, "z");
            source.intensity_cps_1m = ParseDouble(fields, "intensity_cps_1m");
            scene.sources.push_back(source);
        } else if (tokens[0] == "VOLUME") {
            VolumeSpec volume;
            volume.path = ParseString(fields, "path");
            volume.shape = ParseString(fields, "shape");
            volume.tx = ParseDouble(fields, "tx");
            volume.ty = ParseDouble(fields, "ty");
            volume.tz = ParseDouble(fields, "tz");
            volume.qw = ParseDouble(fields, "qw", 1.0);
            volume.qx = ParseDouble(fields, "qx", 0.0);
            volume.qy = ParseDouble(fields, "qy", 0.0);
            volume.qz = ParseDouble(fields, "qz", 0.0);
            volume.sx = ParseDouble(fields, "sx", -1.0);
            volume.sy = ParseDouble(fields, "sy", -1.0);
            volume.sz = ParseDouble(fields, "sz", -1.0);
            volume.radius_m = ParseDouble(fields, "radius_m", -1.0);
            volume.material.name = ParseString(fields, "material_name");
            volume.material.density_g_cm3 = ParseDouble(fields, "density_g_cm3", -1.0);
            volume.material.preset_name = ParseString(fields, "preset_name");
            volume.transport_group = ToLower(ParseString(fields, "transport_group"));
            volume.transport_mode = ToLower(ParseString(fields, "transport_mode", "geant4"));
            volume_index_by_path[volume.path] = scene.volumes.size();
            scene.volumes.push_back(volume);
        } else if (tokens[0] == "COMP") {
            const auto path = ParseString(fields, "path");
            const auto element = ParseString(fields, "element");
            const auto fraction = ParseDouble(fields, "fraction");
            const auto it = volume_index_by_path.find(path);
            if (it != volume_index_by_path.end()) {
                scene.volumes[it->second].material.composition_by_mass[element] = fraction;
            } else if (scene.fe_shield.path == path) {
                scene.fe_shield.material.composition_by_mass[element] = fraction;
            } else if (scene.pb_shield.path == path) {
                scene.pb_shield.material.composition_by_mass[element] = fraction;
            }
        } else if (tokens[0] == "TRI") {
            const auto path = ParseString(fields, "path");
            const auto it = volume_index_by_path.find(path);
            if (it == volume_index_by_path.end()) {
                continue;
            }
            scene.volumes[it->second].triangles.push_back({
                ParseDouble(fields, "ax"), ParseDouble(fields, "ay"), ParseDouble(fields, "az"),
                ParseDouble(fields, "bx"), ParseDouble(fields, "by"), ParseDouble(fields, "bz"),
                ParseDouble(fields, "cx"), ParseDouble(fields, "cy"), ParseDouble(fields, "cz")
            });
        }
    }
    return scene;
}

RequestSpec ReadRequestFile(const std::string& request_path) {
    std::ifstream input(request_path);
    if (!input) {
        throw std::runtime_error("Failed to open request file: " + request_path);
    }
    RequestSpec request;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto tokens = Split(line);
        if (tokens.empty()) {
            continue;
        }
        const auto fields = ParseFields(tokens);
        if (tokens[0] == "STEP") {
            request.step_id = static_cast<int>(ParseLong(fields, "step_id", 0));
            request.dwell_time_s = ParseDouble(fields, "dwell_time_s", 1.0);
            request.seed = ParseLong(fields, "seed", 123);
        } else if (tokens[0] == "POSE") {
            PoseSpec pose;
            pose.x = ParseDouble(fields, "x");
            pose.y = ParseDouble(fields, "y");
            pose.z = ParseDouble(fields, "z");
            pose.qw = ParseDouble(fields, "qw", 1.0);
            pose.qx = ParseDouble(fields, "qx", 0.0);
            pose.qy = ParseDouble(fields, "qy", 0.0);
            pose.qz = ParseDouble(fields, "qz", 0.0);
            const auto kind = ParseString(fields, "kind");
            if (kind == "detector") {
                request.detector_pose = pose;
            } else if (kind == "fe") {
                request.fe_pose = pose;
            } else if (kind == "pb") {
                request.pb_pose = pose;
            }
        }
    }
    return request;
}

std::string GeometryCacheKey(
    const SceneSpec& scene,
    const RequestSpec& request,
    const std::string& physics_profile,
    const int thread_count,
    const std::string& detector_scoring_mode,
    const std::string& secondary_transport_mode
) {
    std::ostringstream stream;
    stream << std::setprecision(17)
           << scene.scene_hash << "|"
           << physics_profile << "|"
           << std::max(1, thread_count) << "|"
           << NormalizeDetectorScoringMode(detector_scoring_mode) << "|"
           << NormalizeSecondaryTransportMode(secondary_transport_mode) << "|"
           << request.detector_pose.x << "," << request.detector_pose.y << "," << request.detector_pose.z << ","
           << request.detector_pose.qw << "," << request.detector_pose.qx << ","
           << request.detector_pose.qy << "," << request.detector_pose.qz << "|"
           << request.fe_pose.x << "," << request.fe_pose.y << "," << request.fe_pose.z << "|"
           << request.pb_pose.x << "," << request.pb_pose.y << "," << request.pb_pose.z;
    return stream.str();
}

class TransportSession {
public:
    TransportSession(
        SceneSpec scene,
        RequestSpec geometry_request,
        std::string physics_profile,
        const int thread_count,
        std::string detector_scoring_mode,
        std::string secondary_transport_mode
    ) : scene_(std::move(scene)),
        geometry_request_(geometry_request),
        physics_profile_(std::move(physics_profile)),
        detector_scoring_mode_(NormalizeDetectorScoringMode(detector_scoring_mode)),
        secondary_transport_mode_(NormalizeSecondaryTransportMode(secondary_transport_mode)),
        thread_count_(std::max(1, thread_count)),
        use_theory_tvl_(UseTheoryTvlProfile(physics_profile_)) {
        for (const auto& volume : scene_.volumes) {
            if (ToLower(volume.transport_mode) != "absorber") {
                continue;
            }
            absorbing_volume_names_.insert(volume.path);
            if (!volume.transport_group.empty()) {
                absorbing_transport_groups_.insert(volume.transport_group);
            }
        }
        run_manager_ = CreateConfiguredRunManager(thread_count_, &run_manager_multithreaded_);
        auto detector = std::make_unique<Geant4SceneConstruction>(
            &scene_,
            &geometry_request_,
            &event_store_,
            &diagnostics_,
            detector_scoring_mode_,
            !use_theory_tvl_
        );
        detector_construction_ = detector.get();
        run_manager_->SetUserInitialization(detector.release());
        G4PhysListFactory factory;
        auto* physics_list = factory.GetReferencePhysList("FTFP_BERT");
        physics_list->ReplacePhysics(new G4EmStandardPhysics_option4());
        run_manager_->SetUserInitialization(physics_list);
        const G4ThreeVector detector_center(
            geometry_request_.detector_pose.x * m,
            geometry_request_.detector_pose.y * m,
            geometry_request_.detector_pose.z * m
        );
        const double detector_radius = DetectorTargetRadiusM(scene_.detector) * m;
        auto action_initialization = std::make_unique<SidecarActionInitialization>(
            &primary_state_,
            absorbing_volume_names_,
            &diagnostics_,
            &event_store_,
            detector_scoring_mode_,
            secondary_transport_mode_,
            detector_center,
            detector_radius
        );
        run_manager_->SetUserInitialization(action_initialization.release());
        run_manager_->Initialize();
    }

    ~TransportSession() {
        delete run_manager_;
    }

    TransportSession(const TransportSession&) = delete;
    TransportSession& operator=(const TransportSession&) = delete;

    SimulationResult Run(
        const RequestSpec& request,
        const double dead_time_tau_s,
        const TransportOptions& options,
        const bool geometry_cache_hit,
        const bool persistent_process
    ) {
        CLHEP::HepRandom::setTheSeed(request.seed);
        if (detector_construction_ != nullptr) {
            detector_construction_->UpdateShieldPoses(request);
        }
        event_store_.ClearDeposits();
        diagnostics_.Clear();
        std::mt19937_64 rng(static_cast<std::uint64_t>(request.seed));
        std::vector<EnergyDeposit> energy_deposits;
        const auto start_time = std::chrono::steady_clock::now();
        long total_primaries = 0;
        double expected_physical_primaries = 0.0;
        double expected_sampled_primaries = 0.0;
        std::map<std::string, double> source_equivalent_counts;
        std::map<std::string, double> transport_detected_counts;
        std::map<std::string, double> transport_uncollided_primary_counts;
        std::map<std::string, double> transport_interacted_primary_counts;
        std::map<std::string, double> transport_secondary_counts;
        const double reference_acceptance = DetectorReferenceAcceptance(scene_.detector);
        const std::string source_rate_model = NormalizeSourceRateModel(options.source_rate_model);
        const bool detector_cps_rate_model = source_rate_model == "detector_cps_1m";
        const bool weighted_transport = !detector_cps_rate_model && UsesSourceBias(options);
        const std::string source_bias_mode = NormalizeSourceBiasMode(options.source_bias_mode);
        const std::string effective_source_bias_mode = detector_cps_rate_model
            ? "detector_cone"
            : source_bias_mode;
        const bool cone_sampled_transport = (
            weighted_transport || effective_source_bias_mode == "detector_cone"
        );
        const double primary_sampling_fraction = std::clamp(
            options.primary_sampling_fraction,
            1.0e-6,
            1.0
        );
        const double primary_history_weight = 1.0 / primary_sampling_fraction;
        double effective_cone_min_deg = std::numeric_limits<double>::infinity();
        double effective_cone_max_deg = 0.0;
        const G4ThreeVector detector_center(
            request.detector_pose.x * m,
            request.detector_pose.y * m,
            request.detector_pose.z * m
        );
        std::map<std::string, double> source_equivalent_counts_by_source;
        std::map<std::string, double> source_equivalent_counts_by_line;
        std::map<std::string, double> transport_detected_counts_by_source;
        std::map<std::string, double> transport_uncollided_primary_counts_by_source;
        std::map<std::string, double> transport_interacted_primary_counts_by_source;
        std::map<std::string, double> transport_secondary_counts_by_source;
        std::map<std::string, double> transport_detected_counts_by_line;
        std::map<std::string, double> transport_uncollided_primary_counts_by_line;
        std::map<std::string, double> transport_interacted_primary_counts_by_line;
        std::map<std::string, double> transport_secondary_counts_by_line;
        for (std::size_t source_index = 0; source_index < scene_.sources.size(); ++source_index) {
            const auto& source = scene_.sources[source_index];
            const std::string source_token = "src" + std::to_string(source_index)
                + "_" + SanitizeMetadataToken(source.isotope);
            const double point_geom_scale = InverseSquareScale(
                source.x,
                source.y,
                source.z,
                request.detector_pose.x,
                request.detector_pose.y,
                request.detector_pose.z
            );
            const double detector_cps_geom_scale = DetectorCpsGeometryScale(
                source.x,
                source.y,
                source.z,
                request.detector_pose.x,
                request.detector_pose.y,
                request.detector_pose.z,
                scene_.detector
            );
            const double geom_scale = detector_cps_rate_model
                ? detector_cps_geom_scale
                : point_geom_scale;
            const auto lines = GammaLinesForIsotope(source.isotope);
            const double shield_transmission = use_theory_tvl_
                ? TheoryTvlTransmission(source, scene_, request)
                : 1.0;
            source_equivalent_counts[source.isotope] += source.intensity_cps_1m
                * request.dwell_time_s
                * geom_scale
                * shield_transmission;
            source_equivalent_counts_by_source[source_token] += source.intensity_cps_1m
                * request.dwell_time_s
                * geom_scale
                * shield_transmission;
            double total_line_intensity = 0.0;
            for (const auto& line : lines) {
                total_line_intensity += std::max(0.0, line.intensity);
            }
            for (const auto& line : lines) {
                const double source_rate_scale = detector_cps_rate_model
                    ? geom_scale
                    : 1.0 / reference_acceptance;
                const double line_weight = (
                    detector_cps_rate_model && total_line_intensity > 0.0
                )
                    ? line.intensity / total_line_intensity
                    : line.intensity;
                const double mean_events = source.intensity_cps_1m
                    * request.dwell_time_s
                    * shield_transmission
                    * line_weight
                    * source_rate_scale;
                const std::string line_token = source_token
                    + "_" + EnergyMetadataToken(line.energy_keV);
                source_equivalent_counts_by_line[line_token] += mean_events;
                if (mean_events <= 0.0) {
                    continue;
                }
                expected_physical_primaries += mean_events;
                const double sampled_mean_events = mean_events * primary_sampling_fraction;
                expected_sampled_primaries += sampled_mean_events;
                std::poisson_distribution<long> distribution(sampled_mean_events);
                const long histories = std::max(0L, distribution(rng));
                if (histories <= 0) {
                    continue;
                }
                total_primaries += histories;
                const double cone_half_angle_rad = EffectiveConeHalfAngleRad(
                    source,
                    scene_,
                    request,
                    options
                );
                if (cone_sampled_transport) {
                    const double cone_half_angle_deg = cone_half_angle_rad * 180.0 / CLHEP::pi;
                    effective_cone_min_deg = std::min(effective_cone_min_deg, cone_half_angle_deg);
                    effective_cone_max_deg = std::max(effective_cone_max_deg, cone_half_angle_deg);
                }
                primary_state_.Configure(
                    G4ThreeVector(source.x * m, source.y * m, source.z * m),
                    line.energy_keV,
                    effective_source_bias_mode,
                    source_rate_model,
                    detector_center,
                    cone_half_angle_rad,
                    weighted_transport ? options.source_bias_isotropic_fraction : 1.0,
                    primary_history_weight
                );
                const auto deposit_start = event_store_.EventDepositsMeV().size();
                BeamOnHistories(run_manager_, histories);
                const auto& deposits = event_store_.EventDepositsMeV();
                energy_deposits.reserve(energy_deposits.size() + deposits.size() - deposit_start);
                for (std::size_t index = deposit_start; index < deposits.size(); ++index) {
                    if (deposits[index].edep_mev <= 0.0) {
                        continue;
                    }
                    const double detected_weight = std::max(0.0, deposits[index].weight);
                    transport_detected_counts[source.isotope] += detected_weight;
                    transport_detected_counts_by_source[source_token] += detected_weight;
                    transport_detected_counts_by_line[line_token] += detected_weight;
                    if (deposits[index].entry_class == DetectorEntryClass::kUncollidedPrimary) {
                        transport_uncollided_primary_counts[source.isotope] += detected_weight;
                        transport_uncollided_primary_counts_by_source[source_token] += detected_weight;
                        transport_uncollided_primary_counts_by_line[line_token] += detected_weight;
                    } else if (deposits[index].entry_class == DetectorEntryClass::kInteractedPrimary) {
                        transport_interacted_primary_counts[source.isotope] += detected_weight;
                        transport_interacted_primary_counts_by_source[source_token] += detected_weight;
                        transport_interacted_primary_counts_by_line[line_token] += detected_weight;
                    } else {
                        transport_secondary_counts[source.isotope] += detected_weight;
                        transport_secondary_counts_by_source[source_token] += detected_weight;
                        transport_secondary_counts_by_line[line_token] += detected_weight;
                    }
                    energy_deposits.push_back({
                        deposits[index].edep_mev * 1000.0,
                        detected_weight
                    });
                }
            }
        }
        constexpr double kBinWidthKeV = 2.0;
        constexpr double kEnergyMaxKeV = 1500.0;
        const int num_bins = static_cast<int>(kEnergyMaxKeV / kBinWidthKeV) + 1;
        std::vector<double> spectrum(num_bins, 0.0);
        std::vector<double> spectrum_variance(num_bins, 0.0);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        const bool incident_gamma_scoring = detector_scoring_mode_ == "incident_gamma_energy";
        for (const auto& deposit : energy_deposits) {
            const double energy_keV = deposit.energy_keV;
            const double smeared = incident_gamma_scoring
                ? energy_keV
                : energy_keV + SigmaEnergyKeV(energy_keV) * gaussian(rng);
            if (smeared < 0.0 || smeared > kEnergyMaxKeV) {
                continue;
            }
            const int index = static_cast<int>(std::floor(smeared / kBinWidthKeV));
            if (index >= 0 && index < num_bins) {
                spectrum[index] += deposit.weight;
                spectrum_variance[index] += deposit.weight * deposit.weight;
            }
        }
        AddBackgroundSpectrum(spectrum, &spectrum_variance, kBinWidthKeV, request.dwell_time_s, options, rng);
        const double total_counts = std::accumulate(spectrum.begin(), spectrum.end(), 0.0);
        const double dwell_time_s = std::max(1.0e-6, request.dwell_time_s);
        const double true_rate = total_counts / dwell_time_s;
        const double observed_scale = 1.0 / (1.0 + std::max(0.0, true_rate * dead_time_tau_s));
        for (std::size_t index = 0; index < spectrum.size(); ++index) {
            spectrum[index] *= observed_scale;
            spectrum_variance[index] *= observed_scale * observed_scale;
        }
        for (auto& item : transport_detected_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_uncollided_primary_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_interacted_primary_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_secondary_counts) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_detected_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_uncollided_primary_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_interacted_primary_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_secondary_counts_by_source) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_detected_counts_by_line) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_uncollided_primary_counts_by_line) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_interacted_primary_counts_by_line) {
            item.second *= observed_scale;
        }
        for (auto& item : transport_secondary_counts_by_line) {
            item.second *= observed_scale;
        }
        const double total_variance = std::accumulate(
            spectrum_variance.begin(),
            spectrum_variance.end(),
            0.0
        );
        const double observed_total_counts = std::accumulate(spectrum.begin(), spectrum.end(), 0.0);
        const double effective_spectrum_entries = total_variance > 0.0
            ? (observed_total_counts * observed_total_counts) / total_variance
            : 0.0;
        const auto end_time = std::chrono::steady_clock::now();
        const auto runtime_s = std::chrono::duration<double>(end_time - start_time).count();
        const double safe_runtime_s = std::max(1.0e-12, runtime_s);
        const auto process_counts = diagnostics_.ProcessCounts();
        const auto volume_step_counts = diagnostics_.VolumeStepCounts();
        const long long compton_count = diagnostics_.ProcessCountForAliases({"compt", "compton"});
        const long long rayleigh_count = diagnostics_.ProcessCountForAliases({"rayl", "rayleigh"});
        const long long photoelectric_count = diagnostics_.ProcessCountForAliases({"phot", "photoelectric"});
        event_store_.ClearDeposits();

        SimulationResult result;
        result.spectrum_counts = std::move(spectrum);
        result.spectrum_count_variance = std::move(spectrum_variance);
        result.metadata["backend"] = "geant4";
        result.metadata["engine_mode"] = "external";
        result.metadata["emission_model"] = detector_cps_rate_model
            ? "detector_equivalent_cone"
            : (weighted_transport ? "weighted_isotropic" : "isotropic");
        result.metadata["source_rate_model"] = source_rate_model;
        result.metadata["intensity_cps_1m_definition"] = "net_detector_count_rate_at_1m";
        result.metadata["line_intensities_normalized"] = detector_cps_rate_model ? "true" : "false";
        result.metadata["physics_profile"] = physics_profile_;
        result.metadata["detector_scoring_mode"] = detector_scoring_mode_;
        result.metadata["detector_fast_scoring"] = detector_scoring_mode_ == "incident_gamma_energy" ? "true" : "false";
        result.metadata["detector_fast_scoring_volume"] = detector_scoring_mode_ == "incident_gamma_energy"
            ? "detector_assembly_entry"
            : "";
        result.metadata["detector_response_applied_in_native"] = detector_scoring_mode_ == "incident_gamma_energy" ? "false" : "true";
        AddGeant4MaterialMuMetadata(result, scene_);
        result.metadata["secondary_transport_mode"] = secondary_transport_mode_;
        result.metadata["gamma_only_secondary_transport"] = secondary_transport_mode_ == "gamma_only" ? "true" : "false";
        result.metadata["theory_tvl_attenuation"] = use_theory_tvl_ ? "true" : "false";
        result.metadata["scene_hash"] = scene_.scene_hash;
        result.metadata["num_primaries"] = std::to_string(total_primaries);
        result.metadata["expected_physical_primaries"] = std::to_string(expected_physical_primaries);
        result.metadata["expected_detector_equivalent_primaries"] = std::to_string(
            expected_physical_primaries
        );
        result.metadata["expected_sampled_primaries"] = std::to_string(expected_sampled_primaries);
        result.metadata["primary_sampling_fraction"] = std::to_string(primary_sampling_fraction);
        result.metadata["primary_history_weight"] = std::to_string(primary_history_weight);
        result.metadata["reference_detector_acceptance"] = std::to_string(reference_acceptance);
        result.metadata["detector_crystal_radius_m"] = std::to_string(scene_.detector.crystal_radius_m);
        result.metadata["detector_housing_thickness_m"] = std::to_string(scene_.detector.housing_thickness_m);
        result.metadata["detector_target_radius_m"] = std::to_string(DetectorTargetRadiusM(scene_.detector));
        const auto fe_shield_normal = ShieldNormalFromPose(request.fe_pose);
        const auto pb_shield_normal = ShieldNormalFromPose(request.pb_pose);
        result.metadata["fe_shield_normal_x"] = std::to_string(fe_shield_normal[0]);
        result.metadata["fe_shield_normal_y"] = std::to_string(fe_shield_normal[1]);
        result.metadata["fe_shield_normal_z"] = std::to_string(fe_shield_normal[2]);
        result.metadata["pb_shield_normal_x"] = std::to_string(pb_shield_normal[0]);
        result.metadata["pb_shield_normal_y"] = std::to_string(pb_shield_normal[1]);
        result.metadata["pb_shield_normal_z"] = std::to_string(pb_shield_normal[2]);
        AddShieldAxisMetadata(result, "fe", request.fe_pose);
        AddShieldAxisMetadata(result, "pb", request.pb_pose);
        result.metadata["total_spectrum_counts"] = std::to_string(observed_total_counts);
        result.metadata["primaries_per_sec"] = std::to_string(static_cast<double>(total_primaries) / safe_runtime_s);
        result.metadata["effective_entries_per_sec"] = std::to_string(effective_spectrum_entries / safe_runtime_s);
        result.metadata["total_track_steps"] = std::to_string(diagnostics_.TotalTrackSteps());
        result.metadata["detector_hit_events"] = std::to_string(energy_deposits.size());
        result.metadata["detector_hit_steps"] = std::to_string(diagnostics_.DetectorHitSteps());
        result.metadata["secondary_count"] = std::to_string(diagnostics_.SecondaryCount());
        result.metadata["killed_non_gamma_secondary_count"] = std::to_string(
            diagnostics_.KilledNonGammaSecondaryCount()
        );
        result.metadata["process_count_compton"] = std::to_string(compton_count);
        result.metadata["process_count_rayleigh"] = std::to_string(rayleigh_count);
        result.metadata["process_count_photoelectric"] = std::to_string(photoelectric_count);
        result.metadata["transport_process_counts"] = SerializeCounterMap(process_counts);
        result.metadata["volume_step_counts"] = SerializeCounterMap(volume_step_counts);
        result.metadata["volume_step_counts_top"] = SerializeTopCounterMap(volume_step_counts, 20);
        result.metadata["requested_threads"] = std::to_string(thread_count_);
        result.metadata["multithreaded_run_manager"] = run_manager_multithreaded_ ? "true" : "false";
        result.metadata["background_cps"] = std::to_string(options.background_cps);
        result.metadata["poisson_background"] = "true";
        result.metadata["weighted_transport"] = weighted_transport ? "true" : "false";
        result.metadata["source_bias"] = effective_source_bias_mode;
        result.metadata["source_bias_mode"] = effective_source_bias_mode;
        result.metadata["source_bias_isotropic_fraction"] = std::to_string(
            weighted_transport ? std::clamp(options.source_bias_isotropic_fraction, 1.0e-6, 1.0) : 1.0
        );
        result.metadata["source_bias_configured_cone_half_angle_deg"] = std::to_string(
            std::max(0.0, options.source_bias_cone_half_angle_deg)
        );
        result.metadata["source_bias_cone_half_angle_deg"] = std::to_string(
            cone_sampled_transport && std::isfinite(effective_cone_max_deg) ? effective_cone_max_deg : 0.0
        );
        result.metadata["cone_half_angle_deg"] = result.metadata["source_bias_cone_half_angle_deg"];
        result.metadata["isotropic_mixture_fraction"] = result.metadata["source_bias_isotropic_fraction"];
        result.metadata["source_bias_effective_cone_half_angle_deg_min"] = std::to_string(
            cone_sampled_transport && std::isfinite(effective_cone_min_deg) ? effective_cone_min_deg : 0.0
        );
        result.metadata["source_bias_effective_cone_half_angle_deg_max"] = std::to_string(
            cone_sampled_transport && std::isfinite(effective_cone_max_deg) ? effective_cone_max_deg : 0.0
        );
        result.metadata["weighted_spectrum_sumw2"] = std::to_string(total_variance);
        result.metadata["weighted_spectrum_effective_entries"] = std::to_string(effective_spectrum_entries);
        result.metadata["absorbing_volume_count"] = std::to_string(absorbing_volume_names_.size());
        result.metadata["absorbing_transport_groups"] = JoinSet(absorbing_transport_groups_, ",");
        result.metadata["persistent_process"] = persistent_process ? "true" : "false";
        result.metadata["geometry_cache_hit"] = geometry_cache_hit ? "true" : "false";
        result.metadata["run_time_s"] = std::to_string(runtime_s);
        for (const auto& item : source_equivalent_counts) {
            result.metadata["source_equivalent_counts_" + item.first] = std::to_string(item.second);
        }
        for (const auto& item : source_equivalent_counts_by_source) {
            result.metadata["source_equivalent_counts_" + item.first] = std::to_string(item.second);
        }
        for (const auto& item : source_equivalent_counts_by_line) {
            result.metadata["source_equivalent_counts_" + item.first] = std::to_string(item.second);
        }
        for (const auto& item : transport_detected_counts) {
            result.metadata["transport_detected_counts_" + item.first] = std::to_string(item.second);
            const double total_detected = std::max(0.0, item.second);
            const double uncollided = std::max(
                0.0,
                transport_uncollided_primary_counts[item.first]
            );
            const double interacted = std::max(
                0.0,
                transport_interacted_primary_counts[item.first]
            );
            const double secondary = std::max(0.0, transport_secondary_counts[item.first]);
            result.metadata["transport_uncollided_primary_counts_" + item.first] = std::to_string(uncollided);
            result.metadata["transport_interacted_primary_counts_" + item.first] = std::to_string(interacted);
            result.metadata["transport_secondary_counts_" + item.first] = std::to_string(secondary);
            result.metadata["transport_non_uncollided_fraction_" + item.first] = std::to_string(
                total_detected > 0.0 ? std::clamp((interacted + secondary) / total_detected, 0.0, 1.0) : 0.0
            );
        }
        for (const auto& item : transport_detected_counts_by_source) {
            result.metadata["transport_detected_counts_" + item.first] = std::to_string(item.second);
            const double total_detected = std::max(0.0, item.second);
            const double uncollided = std::max(0.0, transport_uncollided_primary_counts_by_source[item.first]);
            const double interacted = std::max(0.0, transport_interacted_primary_counts_by_source[item.first]);
            const double secondary = std::max(0.0, transport_secondary_counts_by_source[item.first]);
            result.metadata["transport_uncollided_primary_counts_" + item.first] = std::to_string(uncollided);
            result.metadata["transport_interacted_primary_counts_" + item.first] = std::to_string(interacted);
            result.metadata["transport_secondary_counts_" + item.first] = std::to_string(secondary);
            result.metadata["transport_non_uncollided_fraction_" + item.first] = std::to_string(
                total_detected > 0.0 ? std::clamp((interacted + secondary) / total_detected, 0.0, 1.0) : 0.0
            );
        }
        for (const auto& item : transport_detected_counts_by_line) {
            result.metadata["transport_detected_counts_" + item.first] = std::to_string(item.second);
            const double total_detected = std::max(0.0, item.second);
            const double uncollided = std::max(0.0, transport_uncollided_primary_counts_by_line[item.first]);
            const double interacted = std::max(0.0, transport_interacted_primary_counts_by_line[item.first]);
            const double secondary = std::max(0.0, transport_secondary_counts_by_line[item.first]);
            result.metadata["transport_uncollided_primary_counts_" + item.first] = std::to_string(uncollided);
            result.metadata["transport_interacted_primary_counts_" + item.first] = std::to_string(interacted);
            result.metadata["transport_secondary_counts_" + item.first] = std::to_string(secondary);
            result.metadata["transport_non_uncollided_fraction_" + item.first] = std::to_string(
                total_detected > 0.0 ? std::clamp((interacted + secondary) / total_detected, 0.0, 1.0) : 0.0
            );
        }
        if (!scene_.usd_path.empty()) {
            result.metadata["usd_path"] = scene_.usd_path;
        }
        return result;
    }

private:
    SceneSpec scene_;
    RequestSpec geometry_request_;
    std::string physics_profile_;
    std::string detector_scoring_mode_ = "full_transport";
    std::string secondary_transport_mode_ = "full_transport";
    int thread_count_ = 1;
    bool use_theory_tvl_ = false;
    bool run_manager_multithreaded_ = false;
    EventStore event_store_;
    TransportDiagnostics diagnostics_;
    PrimarySourceState primary_state_;
    Geant4SceneConstruction* detector_construction_ = nullptr;
    G4RunManager* run_manager_ = nullptr;
    std::set<std::string> absorbing_volume_names_;
    std::set<std::string> absorbing_transport_groups_;
};

SimulationResult RunTransport(
    const SceneSpec& scene,
    const RequestSpec& request,
    const std::string& physics_profile,
    const int thread_count,
    const double dead_time_tau_s,
    const TransportOptions& options
) {
    TransportSession session(
        scene,
        request,
        physics_profile,
        thread_count,
        options.detector_scoring_mode,
        options.secondary_transport_mode
    );
    return session.Run(request, dead_time_tau_s, options, false, false);
}

void WriteResponseFile(const SimulationResult& result, const std::string& response_path) {
    std::ofstream output(response_path);
    if (!output) {
        throw std::runtime_error("Failed to open response file: " + response_path);
    }
    for (const auto& item : result.metadata) {
        output << "META " << item.first << "=" << item.second << "\n";
    }
    output << std::setprecision(12);
    output << "SPECTRUM ";
    for (std::size_t index = 0; index < result.spectrum_counts.size(); ++index) {
        if (index > 0) {
            output << ",";
        }
        output << result.spectrum_counts[index];
    }
    output << "\n";
    if (!result.spectrum_count_variance.empty()) {
        output << "SPECTRUM_VARIANCE ";
        for (std::size_t index = 0; index < result.spectrum_count_variance.size(); ++index) {
            if (index > 0) {
                output << ",";
            }
            output << result.spectrum_count_variance[index];
        }
        output << "\n";
    }
}

void RunPersistentServer(
    const std::string& physics_profile,
    const int thread_count,
    const double dead_time_tau_s,
    const TransportOptions& options
) {
    std::unique_ptr<TransportSession> session;
    std::string session_key;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            const auto tokens = Split(line);
            if (tokens.empty()) {
                continue;
            }
            if (tokens[0] == "SHUTDOWN" || tokens[0] == "QUIT") {
                std::cout << "SIMBRIDGE_OK shutdown" << std::endl;
                break;
            }
            if (tokens[0] != "RUN") {
                throw std::runtime_error("Unsupported persistent command: " + tokens[0]);
            }
            const auto fields = ParseFields(tokens);
            const auto scene_path = ParseString(fields, "scene");
            const auto request_path = ParseString(fields, "request");
            const auto response_path = ParseString(fields, "response");
            if (scene_path.empty() || request_path.empty() || response_path.empty()) {
                throw std::runtime_error("RUN requires scene, request, and response fields.");
            }
            const auto scene = ReadSceneFile(scene_path);
            const auto request = ReadRequestFile(request_path);
            const auto key = GeometryCacheKey(
                scene,
                request,
                physics_profile,
                thread_count,
                options.detector_scoring_mode,
                options.secondary_transport_mode
            );
            const bool geometry_cache_hit = session != nullptr && key == session_key;
            if (!geometry_cache_hit) {
                session = std::make_unique<TransportSession>(
                    scene,
                    request,
                    physics_profile,
                    thread_count,
                    options.detector_scoring_mode,
                    options.secondary_transport_mode
                );
                session_key = key;
            }
            auto result = session->Run(
                request,
                dead_time_tau_s,
                options,
                geometry_cache_hit,
                true
            );
            WriteResponseFile(result, response_path);
            std::cout << "SIMBRIDGE_OK response=" << response_path << std::endl;
        } catch (const std::exception& exc) {
            std::cout << "SIMBRIDGE_ERR " << exc.what() << std::endl;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string scene_path;
        std::string request_path;
        std::string response_path;
        std::string physics_profile = "balanced";
        int thread_count = 1;
        double dead_time_tau_s = 5.813e-9;
        bool persistent = false;
        TransportOptions transport_options;
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--scene" && index + 1 < argc) {
                scene_path = argv[++index];
            } else if (arg == "--request" && index + 1 < argc) {
                request_path = argv[++index];
            } else if (arg == "--response" && index + 1 < argc) {
                response_path = argv[++index];
            } else if (arg == "--physics-profile" && index + 1 < argc) {
                physics_profile = argv[++index];
            } else if (arg == "--threads" && index + 1 < argc) {
                thread_count = std::stoi(argv[++index]);
            } else if (arg == "--dead-time-tau-s" && index + 1 < argc) {
                dead_time_tau_s = std::stod(argv[++index]);
            } else if (arg == "--background-cps" && index + 1 < argc) {
                transport_options.background_cps = std::max(0.0, std::stod(argv[++index]));
            } else if (arg == "--source-rate-model" && index + 1 < argc) {
                transport_options.source_rate_model = NormalizeSourceRateModel(argv[++index]);
            } else if (arg == "--source-bias-mode" && index + 1 < argc) {
                transport_options.source_bias_mode = NormalizeSourceBiasMode(argv[++index]);
            } else if (arg == "--source-bias-cone-half-angle-deg" && index + 1 < argc) {
                transport_options.source_bias_cone_half_angle_deg = std::max(0.0, std::stod(argv[++index]));
            } else if (arg == "--source-bias-isotropic-fraction" && index + 1 < argc) {
                transport_options.source_bias_isotropic_fraction = std::clamp(std::stod(argv[++index]), 1.0e-6, 1.0);
            } else if (arg == "--detector-scoring-mode" && index + 1 < argc) {
                transport_options.detector_scoring_mode = NormalizeDetectorScoringMode(argv[++index]);
            } else if (arg == "--secondary-transport-mode" && index + 1 < argc) {
                transport_options.secondary_transport_mode = NormalizeSecondaryTransportMode(argv[++index]);
            } else if (arg == "--primary-sampling-fraction" && index + 1 < argc) {
                transport_options.primary_sampling_fraction = std::clamp(
                    std::stod(argv[++index]),
                    1.0e-6,
                    1.0
                );
            } else if (arg == "--persistent") {
                persistent = true;
            }
        }
        const auto normalized_source_bias_mode = NormalizeSourceBiasMode(transport_options.source_bias_mode);
        if (
            normalized_source_bias_mode != "analog"
            && normalized_source_bias_mode != "mixture_cone_isotropic"
            && normalized_source_bias_mode != "detector_cone"
        ) {
            throw std::runtime_error("Unsupported source bias mode: " + transport_options.source_bias_mode);
        }
        transport_options.source_bias_mode = normalized_source_bias_mode;
        const auto normalized_source_rate_model = NormalizeSourceRateModel(transport_options.source_rate_model);
        if (
            normalized_source_rate_model != "detector_cps_1m"
            && normalized_source_rate_model != "isotropic_emission_equivalent"
        ) {
            throw std::runtime_error("Unsupported source rate model: " + transport_options.source_rate_model);
        }
        transport_options.source_rate_model = normalized_source_rate_model;
        const auto normalized_detector_scoring_mode = NormalizeDetectorScoringMode(
            transport_options.detector_scoring_mode
        );
        if (
            normalized_detector_scoring_mode != "full_transport"
            && normalized_detector_scoring_mode != "incident_gamma_energy"
        ) {
            throw std::runtime_error(
                "Unsupported detector scoring mode: "
                + transport_options.detector_scoring_mode
            );
        }
        transport_options.detector_scoring_mode = normalized_detector_scoring_mode;
        const auto normalized_secondary_transport_mode = NormalizeSecondaryTransportMode(
            transport_options.secondary_transport_mode
        );
        if (
            normalized_secondary_transport_mode != "full_transport"
            && normalized_secondary_transport_mode != "gamma_only"
        ) {
            throw std::runtime_error(
                "Unsupported secondary transport mode: "
                + transport_options.secondary_transport_mode
            );
        }
        transport_options.secondary_transport_mode = normalized_secondary_transport_mode;
        if (persistent) {
            RunPersistentServer(
                physics_profile,
                thread_count,
                dead_time_tau_s,
                transport_options
            );
            return 0;
        }
        if (scene_path.empty() || request_path.empty() || response_path.empty()) {
            throw std::runtime_error(
                "Usage: geant4_sidecar --scene <path> --request <path> --response <path> "
                "or geant4_sidecar --persistent"
            );
        }
        const auto scene = ReadSceneFile(scene_path);
        const auto request = ReadRequestFile(request_path);
        const auto result = RunTransport(
            scene,
            request,
            physics_profile,
            thread_count,
            dead_time_tau_s,
            transport_options
        );
        WriteResponseFile(result, response_path);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
